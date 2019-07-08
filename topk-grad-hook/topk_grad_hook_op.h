#ifndef CAFFE2_OPERATORS_TOPK_GRAD_HOOK_OP_H_
#define CAFFE2_OPERATORS_TOPK_GRAD_HOOK_OP_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class TopKGradHookOp final : public Operator<Context> {
 public:
  // USE_SIMPLE_CTOR_DTOR(TopKGradHookOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  TopKGradHookOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "k", k_, -1) {
    CAFFE_ENFORCE(k_ >= 1, "k argument must be >= 1");
  }

  ~TopKGradHookOp() {}

  bool RunOnDevice() override;

 private:
  // Input: X, Output: X 
  const int k_;
};

template <typename T, class Context>
class TopKGradHookGradientOp final : public Operator<Context> {
 public:
  // USE_SIMPLE_CTOR_DTOR(TopKGradHookGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  TopKGradHookGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "k", k_, -1) {
    CAFFE_ENFORCE(k_ >= 1, "k argument must be >= 1");
  }

  ~TopKGradHookGradientOp() {}

  bool RunOnDevice() override;

 private:
  // Input: dY, Output: dY_topk
  const int k_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TOPK_GRAD_HOOK_OP_H_

