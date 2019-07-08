#include "caffe2/operators/topk_channels_op.h"

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

// __global__ void GradsReduction2D(
//         const int N, 
//         const int C, 
//         const float* dY_data, 
//         float* dY_reduction_data) {
//     /// 4D NxCxHxW grads tensor reduction to 2D NxC grads tensor
//     CUDA_1D_KERNEL_LOOP(i, N * C) {
//         dY_reduction_data[i] = fabsf(dY_data[i]);
//     }
// }

__global__ void ChannelZeroOut4D(
        const int N, 
        const int C, 
        const int k,
        const int H, 
        const int W,
        const float* dY_data, 
        float* dY_topk_data, 
        const TIndex* indices_data) {
    /// zero out specific channel dimention of 4D NxCxHxW
    CUDA_1D_KERNEL_LOOP(i, N * k) {
        int ii = i / k;
        int zero_channel = indices_data[i];
        int begin_offset = ii * (C * H * W) + zero_channel * (H * W);
        memcpy(dY_topk_data + begin_offset,
                dY_data + begin_offset, 
                (H * W) * sizeof(float));
    }
}

// __global__ void ChannelZeroOut2D(
//         const int N, 
//         const int C, 
//         const int k,
//         const float* dY_data, 
//         float* dY_topk_data, 
//         const TIndex* indices_data) {
//     /// zero out specific channel dimention of 4D NxCxHxW
//     CUDA_1D_KERNEL_LOOP(i, N * k) {
//         int ii = i / k;
//         int zero_channel = indices_data[i];
//         dY_topk_data[ii * C + zero_channel] = dY_data[ii * C + zero_channel];
//     }
// }

} // namespace


template <typename T>
class TopKChannelsOp<T, CUDAContext> : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  TopKChannelsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        OP_SINGLE_ARG(int, "k", k_, -1) {
    CAFFE_ENFORCE(k_ >= 1, "k argument must be >= 1");
  }

  ~TopKChannelsOp(){};

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
bool TopKChannelsOp<T, CUDAContext>::RunOnDevice() {
    /// #1 --- input check
    const auto& X = Input(0);
    auto* Y = Output(0);
    auto* Mask = Output(1);
    const std::vector<TIndex>& input_dims = X.dims();
    // The input tensor must be 4D tensor
    CAFFE_ENFORCE(X.ndim() == 4, "The dimession of input tensor must be 4");
    Y->ResizeLike(X);


    /// #2 --- reduce X back to (NxC) tensor by summing the last two dimentions
    int N = X.dim32(0);    // batchsize
    int C = X.dim32(1);    // channel
    std::vector<TIndex> output_dims = {N, C};
    Tensor<CUDAContext> X_reduction(output_dims);

    const float* X_original_data = X.template data<float>();
    int H = X.dim32(2);
    int W = X.dim32(3);
    float* X_reduction_data = X_reduction.template mutable_data<float>();
    GradsReduction4D<<<
        CAFFE_GET_BLOCKS(X_reduction.size()),
        CAFFE_CUDA_NUM_THREADS, 
        0, context_.cuda_stream()>>>( 
                N, C, H, W, 
                X_original_data, 
                X_reduction.template mutable_data<float>());


    /// #3 --- get topk in channel level
    int AXIS_ = 1;  // hard-coded, only for channel dimension topk
    CAFFE_ENFORCE_LE(
        k_,
        output_dims[AXIS_],
        "k argument should not be greater than the channel dim.");

    std::vector<TIndex> topk_dims = output_dims;
    topk_dims[AXIS_] = k_;
    Tensor<CUDAContext> values(topk_dims);
    Mask->Resize(topk_dims); // allocate the memory for Mask
    float* values_data = values.template mutable_data<float>();
    TIndex* mask_data = Mask->template mutable_data<TIndex>();

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
    const TIndex outer_size = X_reduction.size() / output_dims[AXIS_];
    const TIndex inner_size = output_dims[AXIS_];

    RunTopKOnLastDimCUDAImpl<T>(
        // X_reduction_data,
        X_reduction.template data<float>(),
        outer_size,
        inner_size,
        k_,
        values_data,
        mask_data,
        &context_);


    /// #4 --- zero out channels smaller than topk value
    float* Y_data = Y->template mutable_data<float>();
    ChannelZeroOut4D<<<
        CAFFE_GET_BLOCKS(Mask->size()),
        CAFFE_CUDA_NUM_THREADS, 
        0, context_.cuda_stream()>>>( 
                N, C, k_, H, W, 
                X_original_data, 
                Y_data,
                mask_data);

    return true;
}   //TopKChannelsOp::RunOnDevice


template <>
bool TopKChannelsGradientOp<float, CUDAContext>::RunOnDevice() {
    /// #1 --- input check
    auto& dY = Input(0);
    auto& Mask = Input(1);
    auto* dX = Output(0);
    const std::vector<TIndex>& input_dims = dY.dims();
    // The input gradient tensor must be 4D tensor
    CAFFE_ENFORCE(dY.ndim() == 4, "The dimession of input tensor must be 4");
    dX->ResizeLike(dY);


    /// #2 --- zero out corresponding graddient channels with topk mask 
    int N = dY.dim32(0);    // batchsize
    int C = dY.dim32(1);    // channel
    int H = dY.dim32(2);
    int W = dY.dim32(3);
    CAFFE_ENFORCE_LE(k_, C, "k should be no greater than the channel dim.");

    const float* dY_data = dY.template data<float>();
    const TIndex* mask_data = Mask.template data<TIndex>();
    float* dX_data = dX->template mutable_data<float>();
    ChannelZeroOut4D<<<
        CAFFE_GET_BLOCKS(Mask.size()),
        CAFFE_CUDA_NUM_THREADS, 
        0, context_.cuda_stream()>>>( 
                N, C, k_, H, W, 
                dY_data,
                dX_data,
                mask_data);

    return true;
}   //TopKChannelsGradientOp::RunOnDevice



REGISTER_CUDA_OPERATOR(TopKChannels, TopKChannelsOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(TopKChannelsGradient, 
        TopKChannelsGradientOp<float, CUDAContext>);

} // namespace caffe2
