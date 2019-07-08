#include "caffe2/operators/topk_grad_hook_op.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <utility>
#include <vector>
#include <random>

#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/tensor.h"


namespace caffe2 {

// namespace for helper functions
namespace {
void Transpose() {
}

template <typename T>
struct ValueComp {
  bool operator()(
      const std::pair<T, TIndex>& lhs,
      const std::pair<T, TIndex>& rhs) const {
    return lhs.first > rhs.first ||
        (lhs.first == rhs.first && lhs.second < rhs.second);
  }
};

template <typename T>
void GetTopK(
    const T* input,
    const TIndex n,
    const TIndex k,
    const TIndex src_offset,
    const TIndex dst_offset,
    const TIndex stride,
    T* values,
    TIndex* indices) {
  const T* src_ptr = input + src_offset;
  std::vector<std::pair<T, TIndex>> heap_data;
  heap_data.reserve(k);
  for (TIndex i = 0; i < k; ++i) {
    heap_data.emplace_back(*src_ptr, i);
    src_ptr += stride;
  }
  std::priority_queue<
      std::pair<T, TIndex>,
      std::vector<std::pair<T, TIndex>>,
      ValueComp<T>>
      pq(ValueComp<T>(), std::move(heap_data));
  for (TIndex i = k; i < n; ++i) {
    if (pq.top().first < *src_ptr) {
      pq.pop();
      pq.emplace(*src_ptr, i);
    }
    src_ptr += stride;
  }
  TIndex dst_pos = dst_offset + (k - 1) * stride;
  while (!pq.empty()) {
    const auto& item = pq.top();
    values[dst_pos] = item.first;
    indices[dst_pos] = item.second;
    pq.pop();
    dst_pos -= stride;
  }
}

template <typename T>
T value_abs(T v) {
    return v >= 0 ? v : -v;
}
} // namespace


template <>
bool TopKGradHookOp<float, CPUContext>::RunOnDevice() {
  // input check
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  Y->CopyFrom(X);

  return true;
}   //TopKGradHookOp::RunOnDevice

template <>
bool TopKGradHookGradientOp<float, CPUContext>::RunOnDevice() {
    std::cout << "===================== In topk gradient =======================" << std::endl;

    // input check
    auto& dY = Input(0);
    auto* dY_topk = Output(0);
    const std::vector<TIndex>& input_dims = dY.dims();
    // The input tensor must be 4D or 2D tensor
    CAFFE_ENFORCE(dY.ndim() == 2 || dY.ndim() == 4,
            "The dimession of input tensor must be 2 or 4");
    dY_topk->ResizeLike(dY);

    
    // if input is 4D tensor (NxCxHxW), reduce it back to (NxC) tensor by
    // summing the last two dimentions
    int N = dY.dim32(0);    // batchsize
    int C = dY.dim32(1);    // channel
    std::vector<TIndex> output_dims = {N, C};
    Tensor<CPUContext> dY_reduction(output_dims);

    const float* dY_original_data = dY.template data<float>();
    if (dY.ndim() == 4) {
        int H = dY.dim32(2);
        int W = dY.dim32(3);
        float* dY_reduction_data = dY_reduction.template mutable_data<float>();
        for (TIndex i = 0; i < N; ++i) {
            for (TIndex j = 0; j < C; ++j) { 
                float tmp_sum = 0.;
                for (TIndex h = 0; h < H; ++h) {
                    for (TIndex w = 0; w < W; ++ w) {
                        tmp_sum += value_abs(dY_original_data[i * (C * H * W) 
                                + j * (H * W) + h * W + w]);
                    }
                }
                dY_reduction_data[i * C + j] = tmp_sum;
            }
        }
    }else {
        // dY_reduction.CopyFrom(dY);
        float* dY_reduction_data = dY_reduction.template mutable_data<float>();
        for (TIndex i = 0; i < N; ++i) {
            for (TIndex j = 0; j < C; ++j) { 
                dY_reduction_data[i * C + j] = value_abs(
                        dY_original_data[i * C + j]);
            }
        }
    }

    // DEBUG PRINT
    std::cout << "dY reduction data (only first two dimensions)\n";
    float* dY_reduction_data = dY_reduction.template mutable_data<float>();
    for (TIndex i = 0; i < N; ++i) {
       for (TIndex j = 0; j < C; ++j) {
           std::cout << dY_reduction_data[i * C + j] << ", ";
       }
       std::cout << std::endl;
    }
    std::cout << "print over" << std::endl;
    
    // get topk in channel level
    int AXIS_ = 1;
    CAFFE_ENFORCE_LE(
        k_,
        output_dims[AXIS_],
        "k argument should not be greater than the channel dim.");

    std::vector<TIndex> topk_dims = output_dims;
    topk_dims[AXIS_] = k_;
    Tensor<CPUContext> values(topk_dims);
    Tensor<CPUContext> indices(topk_dims);
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
    const TIndex src_offset_stride = output_dims[AXIS_] * next_size;
    const TIndex dst_offset_stride = k_ * next_size;
    TIndex src_offset = 0;
    TIndex dst_offset = 0;
    for (TIndex i = 0; i < prev_size; ++i) {
        for (TIndex j = 0; j < next_size; ++j) {
            GetTopK(
                dY_reduction_data,
                output_dims[AXIS_],
                k_,
                src_offset + j,
                dst_offset + j,
                next_size,
                values_data,
                indices_data);
        }
        src_offset += src_offset_stride;
        dst_offset += dst_offset_stride;
    }

    // DEBUG PRINT
    std::cout << "topk values of dY reduction data\n";
    for (TIndex i = 0; i < N; ++i) {
       for (TIndex j = 0; j < k_; ++j) {
           std::cout << values_data[i * k_ + j] << ", ";
       }
       std::cout << std::endl;
    }
    std::cout << "print over" << std::endl;
    
    std::cout << "topk indices of dY reduction data\n";
    for (TIndex i = 0; i < N; ++i) {
       for (TIndex j = 0; j < k_; ++j) {
           std::cout << indices_data[i * k_ + j] << ", ";
       }
       std::cout << std::endl;
    }
    std::cout << "print over" << std::endl;

    // zero out channels smaller than topk value
    float* dY_topk_data = dY_topk->template mutable_data<float>();
    if (dY.ndim() == 4) {
        int H = dY.dim32(2);
        int W = dY.dim32(3);
        for (TIndex i = 0; i < N; ++i) {
            for (TIndex j = 0; j < k_; ++j) { 
                TIndex zero_channel = indices_data[i * k_ + j];
                for (TIndex h = 0; h < H; ++h) {
                    for (TIndex w = 0; w < W; ++ w) {
                        dY_topk_data[i * (C * H * W) + zero_channel * (H * W) + h * W + w] = 
                            dY_original_data[i * (C * H * W) + zero_channel * (H * W) + h * W + w];
                    }
                }
            }
        }
    }else {
        for (TIndex i = 0; i < N; ++i) {
            for (TIndex j = 0; j < k_; ++j) { 
                TIndex zero_channel = indices_data[i * k_ + j];
                dY_topk_data[i * C + zero_channel] = dY_original_data[i * C + zero_channel];
            }
        }
    }

    std::cout << "===================== In topk gradient =======================" << std::endl;
    return true;
}   //TopKGradHookGradientOp::RunOnDevice

REGISTER_CPU_OPERATOR(TopKGradHook, TopKGradHookOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(TopKGradHookGradient, TopKGradHookGradientOp<float, CPUContext>);

// Input: X, output: X
OPERATOR_SCHEMA(TopKGradHook)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
The TopKGradHook operator takes in the input X and generate the output X for 
forward pass, for the backward pass, it change the upstream gradient of dY to
dY_topk, now we simply make those value in dy less than the 'topk'_th value to
be zero(zero out) to validate the idea of meProp.
)DOC")
    .Input(0, "X", "(N x C x H x W) or (N x D) tensor for part of operator output, \
            N denotes batch size")
    .Output(0, "Y", "(N x C x H x W) or (N x D)  tensor for part of operator output, \
            N denotes batch size")
    .Arg("k", "Number of top elements to retrieve");

// Input: dY, output: dY_topk
OPERATOR_SCHEMA(TopKGradHookGradient)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
TopKGradHookGradient modifies dY to get the dY_topk for the donwstream gradient
propogation. Acctually, the TopKGradHookOp + TopKGradHookGradientOp together are
an grad dummpy operator, which is, this operator does not change X in forward
pass, only change the upstream dY in backward pass for computing the 
downstream dY
)DOC");

class GetTopKGradHookGradient: public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
      def_.type() + "Gradient",
      "",
      vector<string>{GO(0)},
      vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(TopKGradHook, GetTopKGradHookGradient);
}  // namespace caffe2

