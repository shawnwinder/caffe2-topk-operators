#include "caffe2/operators/topk_grad_hook_op.h"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/top_k_heap_selection.cuh"
#include "caffe2/operators/top_k_radix_selection.cuh"
#include "caffe2/utils/math.h"



namespace caffe2 {

namespace {

template <typename T, int kHeapSize, bool kSelectMax = true>
void RunHeapSelectionImpl(
    const T* input,
    const TIndex outer_size,
    const TIndex inner_size,
    const int k,
    T* values,
    TIndex* indices,
    CUDAContext* context) {
  constexpr int kBlockSize = 256;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  constexpr int smem = kNumWarps * kHeapSize * (sizeof(T) + sizeof(TIndex));
  constexpr T kInitVal = kSelectMax ? std::numeric_limits<T>::lowest()
                                    : std::numeric_limits<T>::max();
  selectRowsViaHeap<T, TIndex, TIndex, kBlockSize, kHeapSize, kSelectMax>
      <<<outer_size, kBlockSize, smem, context->cuda_stream()>>>(
          input,
          values,
          indices,
          kInitVal,
          std::numeric_limits<TIndex>::max(),
          outer_size,
          inner_size,
          k);
}

template <typename T, bool kSelectMax = true>
void RunRadixSelectionImpl(
    const T* input,
    const TIndex outer_size,
    const TIndex inner_size,
    const int k,
    T* values,
    TIndex* indices,
    CUDAContext* context) {
  const int block = std::min(
      math::roundUp(static_cast<int>(inner_size), kWarpSize),
      CAFFE_CUDA_NUM_THREADS);
  gatherTopK<T, kSelectMax, TIndex>
      <<<outer_size, block, 0, context->cuda_stream()>>>(
          input, inner_size, k, outer_size, values, indices);
  // Unfortunately the output is not currently sorted, and there is no batch
  // sorting utility available. Iterate over all of the slices and sort them
  // in-place using Thrust.
  for (int i = 0; i < outer_size; ++i) {
    thrust::sort_by_key(
        thrust::cuda::par.on(context->cuda_stream()),
        values + i * k,
        values + i * k + k,
        indices + i * k,
        thrust::greater<T>());
  }
}

template <typename T>
void RunTopKOnLastDimCUDAImpl(
    const T* input,
    const TIndex outer_size,
    const TIndex inner_size,
    const int k,
    T* values,
    TIndex* indices,
    CUDAContext* context) {
  // If k is small, uses heap selection, otherwise uses radix selection.
  if (k < 32) {
    RunHeapSelectionImpl<T, 32>(
        input, outer_size, inner_size, k, values, indices, context);
  } else if (k < 128) {
    RunHeapSelectionImpl<T, 128>(
        input, outer_size, inner_size, k, values, indices, context);
  } else if (k < 512) {
    RunHeapSelectionImpl<T, 512>(
        input, outer_size, inner_size, k, values, indices, context);
  } else {
    RunRadixSelectionImpl<T>(
        input, outer_size, inner_size, k, values, indices, context);
  }
}

__global__ void GradsReduction4D(
        const int N, 
        const int C, 
        const int H, 
        const int W,
        const float* dY_data, 
        float* dY_reduction_data) {
    /// 4D NxCxHxW grads tensor reduction to 2D NxC grads tensor
    CUDA_1D_KERNEL_LOOP(i, N * C) {
        int begin_offset = i * H * W;
        int end_offset = (i + 1) * H * W;
        dY_reduction_data[i] = thrust::transform_reduce(
                thrust::device,
                dY_data + begin_offset, 
                dY_data + end_offset,
                fabsf,
                0,
                thrust::plus<float>());
    }
}

__global__ void GradsReduction2D(
        const int N, 
        const int C, 
        const float* dY_data, 
        float* dY_reduction_data) {
    /// 4D NxCxHxW grads tensor reduction to 2D NxC grads tensor
    CUDA_1D_KERNEL_LOOP(i, N * C) {
        dY_reduction_data[i] = fabsf(dY_data[i]);
    }
}

__global__ void ChannelZeroOut4D(
        const int N, 
        const int C, 
        const int k,
        const int H, 
        const int W,
        const float* dY_data, 
        float* dY_topk_data, 
        TIndex* indices_data) {
    /// zero out specific channel dimention of 4D NxCxHxW
    CUDA_1D_KERNEL_LOOP(i, N * k) {
        int ii = i / k;
        // int jj = i % k;
        int zero_channel = indices_data[i];
        int begin_offset = ii * (C * H * W) + zero_channel * (H * W);
        // int end_offset = ii * (C * H * W) + (zero_channel + 1) * (H * W);
        memcpy(dY_topk_data + begin_offset,
                dY_data + begin_offset, 
                (H * W) * sizeof(float));
    }
}

__global__ void ChannelZeroOut2D(
        const int N, 
        const int C, 
        const int k,
        const float* dY_data, 
        float* dY_topk_data, 
        TIndex* indices_data) {
    /// zero out specific channel dimention of 4D NxCxHxW
    CUDA_1D_KERNEL_LOOP(i, N * k) {
        int ii = i / k;
        int zero_channel = indices_data[i];
        dY_topk_data[ii * C + zero_channel] = dY_data[ii * C + zero_channel];
    }
}

} // namespace


template <>
bool TopKGradHookOp<float, CUDAContext>::RunOnDevice() {
  // input check
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  Y->CopyFrom(X);

  return true;
}   //TopKGradHookOp::RunOnDevice


template <typename T>
class TopKGradHookGradientOp<T, CUDAContext> : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  TopKGradHookGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        OP_SINGLE_ARG(int, "k", k_, -1) {
    CAFFE_ENFORCE(k_ >= 1, "k argument must be >= 1");
  }

  ~TopKGradHookGradientOp(){};

  bool RunOnDevice() override;

 private:
  const int k_;

  // Buffers for CUDAContext.
  TensorCUDA input_transposed_buffer_;
  TensorCUDA values_transposed_buffer_;
  TensorCUDA indices_transposed_buffer_;

  // Shape tensors on device for CUDAContext.
  TensorCUDA input_dims_device_;
  TensorCUDA input_transposed_dims_device_;
  TensorCUDA input_axes_device_;

  TensorCUDA output_dims_device_;
  TensorCUDA output_transposed_dims_device_;
  TensorCUDA output_transposed_axes_device_;
};

template <typename T>
bool TopKGradHookGradientOp<T, CUDAContext>::RunOnDevice() {
    std::cout << "========================= In TopKGradHookGradientOp op (GPU) ======================" << std::endl;
    /// #1 --- input check
    const auto& dY = Input(0);
    auto* dY_topk = Output(0);
    const std::vector<TIndex>& input_dims = dY.dims();
    // The input tensor must be 4D or 2D tensor
    CAFFE_ENFORCE(dY.ndim() == 2 || dY.ndim() == 4,
            "The dimession of input tensor must be 2 or 4");
    dY_topk->ResizeLike(dY);


    /// #2 --- if input is 4D tensor (NxCxHxW), reduce it back to 2D tensor 
    /// (NxC) by summing the last two dimentions
    int N = dY.dim32(0);    // batchsize
    int C = dY.dim32(1);    // channel
    std::vector<TIndex> output_dims = {N, C};
    Tensor<CUDAContext> dY_reduction(output_dims);

    const float* dY_original_data = dY.template data<float>();
    if (dY.ndim() == 4) {
        int H = dY.dim32(2);
        int W = dY.dim32(3);
        float* dY_reduction_data = dY_reduction.template mutable_data<float>();
        GradsReduction4D<<<
            CAFFE_GET_BLOCKS(dY_reduction.size()),
            CAFFE_CUDA_NUM_THREADS, 
            0, context_.cuda_stream()>>>( 
                    N, C, H, W, 
                    dY_original_data, 
                    dY_reduction.template mutable_data<float>());
    }else {
        // dY_reduction.CopyFrom(dY);
        GradsReduction2D<<<
            CAFFE_GET_BLOCKS(dY_reduction.size()),
            CAFFE_CUDA_NUM_THREADS, 
            0, context_.cuda_stream()>>>( 
                    N, C,
                    dY_original_data, 
                    dY_reduction.template mutable_data<float>());
    }

    // DEBUG PRINT
    Tensor<CPUContext> dY_reduction_CPU(output_dims);
    context_.Copy<float, CUDAContext, CPUContext>(
            dY_reduction.size(), 
            dY_reduction.data<float>(), 
            dY_reduction_CPU.mutable_data<float>());
    std::cout << "dY reduction data (only first two dimensions)\n";
    const float* dY_reduction_cpu_data = dY_reduction_CPU.data<float>();
    for (TIndex i = 0; i < N; ++i) {
       for (TIndex j = 0; j < C; ++j) {
           std::cout << dY_reduction_cpu_data[i * C + j] << ", ";
       }
       std::cout << std::endl;
    }
    std::cout << "print over" << std::endl;


    /// #3 --- get topk in channel level
    int AXIS_ = 1;  // hard-coded, only for channel dimension topk
    CAFFE_ENFORCE_LE(
        k_,
        output_dims[AXIS_],
        "k argument should not be greater than the channel dim.");

    std::vector<TIndex> topk_dims = output_dims;
    topk_dims[AXIS_] = k_;
    Tensor<CUDAContext> values(topk_dims);
    Tensor<CUDAContext> indices(topk_dims);
    float* values_data = values.template mutable_data<float>();
    TIndex* indices_data = indices.template mutable_data<TIndex>();

    const TIndex prev_size = std::accumulate(
        output_dims.cbegin(),
        output_dims.cbegin() + AXIS_,
        TIndex(1),
        std::multiplies<TIndex>());
    const TIndex next_size = std::accumulate(
        output_dims.cbegin() + AXIS_ + 1,
        output_dims.cend(),
        TIndex(1),
        std::multiplies<TIndex>());
    const TIndex outer_size = dY_reduction.size() / output_dims[AXIS_];
    const TIndex inner_size = output_dims[AXIS_];

    RunTopKOnLastDimCUDAImpl<T>(
        // dY_reduction_data,
        dY_reduction.template data<float>(),
        outer_size,
        inner_size,
        k_,
        values_data,
        indices_data,
        &context_);

    // DEBUG PRINT
    Tensor<CPUContext> values_cpu(topk_dims);
    context_.Copy<float, CUDAContext, CPUContext>(
            values.size(), 
            values.data<float>(), 
            values_cpu.mutable_data<float>());
    std::cout << "dY reduction topk values\n";
    const float* values_cpu_data = values_cpu.data<float>();
    for (TIndex i = 0; i < N; ++i) {
       for (TIndex j = 0; j < k_; ++j) {
           std::cout << values_cpu_data[i * k_ + j] << ", ";
       }
       std::cout << std::endl;
    }
    std::cout << "print over\n" << std::endl;

    Tensor<CPUContext> indices_cpu(topk_dims);
    context_.Copy<TIndex, CUDAContext, CPUContext>(
            indices.size(), 
            indices.data<TIndex>(), 
            indices_cpu.mutable_data<TIndex>());
    std::cout << "dY reduction topk indices\n";
    const TIndex* indices_cpu_data = indices_cpu.data<TIndex>();
    for (TIndex i = 0; i < N; ++i) {
       for (TIndex j = 0; j < k_; ++j) {
           std::cout << indices_cpu_data[i * k_ + j] << ", ";
       }
       std::cout << std::endl;
    }
    std::cout << "print over\n" << std::endl;

    /// #4 --- zero out channels smaller than topk value
    float* dY_topk_data = dY_topk->template mutable_data<float>();
    if (dY.ndim() == 4) {
        int H = dY.dim32(2);
        int W = dY.dim32(3);
        ChannelZeroOut4D<<<
            CAFFE_GET_BLOCKS(indices.size()),
            CAFFE_CUDA_NUM_THREADS, 
            0, context_.cuda_stream()>>>( 
                    N, C, k_, H, W, 
                    dY_original_data, 
                    dY_topk_data,
                    indices_data);
    }else {
        ChannelZeroOut2D<<<
            CAFFE_GET_BLOCKS(indices.size()),
            CAFFE_CUDA_NUM_THREADS, 
            0, context_.cuda_stream()>>>( 
                    N, C, k_,
                    dY_original_data, 
                    dY_topk_data,
                    indices_data);
    }

    std::cout << "========================= In TopKGradHookGradientOp op (GPU) ======================" << std::endl;
    return true;
}

REGISTER_CUDA_OPERATOR(TopKGradHook, TopKGradHookOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(TopKGradHookGradient, 
        TopKGradHookGradientOp<float, CUDAContext>);

} // namespace caffe2
