#ifndef CAFFE2_OPERATORS_TOPK_CHANNELS_OP_H_
#define CAFFE2_OPERATORS_TOPK_CHANNELS_OP_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class TopKChannelsOp final : public Operator<Context> {
 public:
  // USE_SIMPLE_CTOR_DTOR(TopKChannelsOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  TopKChannelsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "k", k_, -1) {
    CAFFE_ENFORCE(k_ >= 1, "k argument must be >= 1");
  }

  ~TopKChannelsOp() {}

  bool RunOnDevice() override;

 private:
  // Input: X, Output: Y, Mask 
  const int k_;
};

template <typename T, class Context>
class TopKChannelsGradientOp final : public Operator<Context> {
 public:
  // USE_SIMPLE_CTOR_DTOR(TopKChannelsGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  TopKChannelsGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "k", k_, -1) {
    CAFFE_ENFORCE(k_ >= 1, "k argument must be >= 1");
  }

  ~TopKChannelsGradientOp() {}

  bool RunOnDevice() override;

 private:
  // Input: dY, Mask, Output: dX(dY_topk)
  const int k_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CHANNELS_OP_H_
