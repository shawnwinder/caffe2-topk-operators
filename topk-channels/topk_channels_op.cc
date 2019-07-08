#include "caffe2/operators/topk_channels_op.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/tensor.h"


namespace caffe2 {

// namespace for helper functions
namespace {

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
bool TopKChannelsOp<float, CPUContext>::RunOnDevice() {
    /// #1 --- input check
    auto& X = Input(0);
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
    Tensor<CPUContext> X_reduction(output_dims);

    const float* X_original_data = X.template data<float>();
    int H = X.dim32(2);
    int W = X.dim32(3);
    float* X_reduction_data = X_reduction.template mutable_data<float>();
    for (TIndex i = 0; i < N; ++i) {
        for (TIndex j = 0; j < C; ++j) { 
            float tmp_sum = 0.;
            for (TIndex h = 0; h < H; ++h) {
                for (TIndex w = 0; w < W; ++ w) {
                    tmp_sum += value_abs(X_original_data[i * (C * H * W) 
                            + j * (H * W) + h * W + w]);
                }
            }
            X_reduction_data[i * C + j] = tmp_sum;
        }
    }

    
    /// #3 --- get topk in channel level
    int AXIS_ = 1;
    CAFFE_ENFORCE_LE(
        k_,
        output_dims[AXIS_],
        "k argument should not be greater than the channel dim.");

    std::vector<TIndex> topk_dims = output_dims;
    topk_dims[AXIS_] = k_;
    Tensor<CPUContext> values(topk_dims);
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
    const TIndex src_offset_stride = output_dims[AXIS_] * next_size;
    const TIndex dst_offset_stride = k_ * next_size;
    TIndex src_offset = 0;
    TIndex dst_offset = 0;
    for (TIndex i = 0; i < prev_size; ++i) {
        for (TIndex j = 0; j < next_size; ++j) {
            GetTopK(
                // X_reduction_data,
                X_reduction.template data<float>(),
                output_dims[AXIS_],
                k_,
                src_offset + j,
                dst_offset + j,
                next_size,
                values_data,
                mask_data);
        }
        src_offset += src_offset_stride;
        dst_offset += dst_offset_stride;
    }


    /// #4 --- zero out channels smaller than topk value
    float* Y_data = Y->template mutable_data<float>();
    for (TIndex i = 0; i < N; ++i) {
        for (TIndex j = 0; j < k_; ++j) { 
            TIndex zero_channel = mask_data[i * k_ + j];
            for (TIndex h = 0; h < H; ++h) {
                for (TIndex w = 0; w < W; ++ w) {
                    Y_data[i * (C * H * W) + zero_channel * (H * W) + h * W + w] = 
                        X_original_data[i * (C * H * W) + zero_channel * (H * W) + h * W + w];
                }
            }
        }
    }

    return true;
}   //TopKChannelsOp::RunOnDevice

template <>
bool TopKChannelsGradientOp<float, CPUContext>::RunOnDevice() {
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
    for (TIndex i = 0; i < N; ++i) {
        for (TIndex j = 0; j < k_; ++j) { 
            TIndex zero_channel = mask_data[i * k_ + j];
            for (TIndex h = 0; h < H; ++h) {
                for (TIndex w = 0; w < W; ++w) {
                    dX_data[i * (C * H * W) + zero_channel * (H * W) + h * W + w] = 
                        dY_data[i * (C * H * W) + zero_channel * (H * W) + h * W + w];
                }
            }
        }
    }

    return true;
}   //TopKChannelsGradientOp::RunOnDevice


REGISTER_CPU_OPERATOR(TopKChannels, TopKChannelsOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(TopKChannelsGradient, TopKChannelsGradientOp<float, CPUContext>);

// Input: X(feature), output: Y(topk fature), Mask(masked indices)
OPERATOR_SCHEMA(TopKChannels)
    .NumInputs(1)
    .NumOutputs(1, 2)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
The TopKChannels operator takes in the input feature, sum elements along each
channel, then select the topk channel of the original feature. This operator
has two outputs, the  topk feature and the corresponding indices, and used in
both training & inference. Temporarily we simply make those unselected feature
channel values to be zero(zero out) to validate the idea, if it works, we will
actually compress the weight params.
)DOC")
    .Input(0, "X", "(N x C x H x W) feature tensor, N denotes batch size")
    .Output(0, "Y", "(N x C x H x W) feature tensor, N denotes batch size")
    .Output(1, "Mask", "(N x D) topk channel indices tensor, N denotes batch size")
    .Arg("k", "Number of top elements to retrieve");

// Input: dY, Mask, output: dX
OPERATOR_SCHEMA(TopKChannelsGradient)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
The gradient of TopKChannels, basically resembles the 'Dropout' op except that 
this op does not distinguish the training or testing phase.
)DOC");

class GetTopKChannelsGradient: public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
      def_.type() + "Gradient",
      "",
      vector<string>{GO(0), O(1)},
      vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(TopKChannels, GetTopKChannelsGradient);
}  // namespace caffe2

