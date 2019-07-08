#include "caffe2/operators/fft_op.h"
#include "caffe2/utils/math.h"

#include "Eigen/Core"
#include "unsupported/Eigen/FFT"


namespace caffe2 {

/*
 * @brief: FFT library knowledge needs to be noticed: 
 * the Eigen::FFT compute the whole spectrum
 * of the fft forward out, which means the output size is the same to the input
 * size for batch 1D FFT. Althought to recover the input, in other words, for 
 * the inverse transformation, we actually only need the fron half of the output
 * complex matrix derived from fft.fwd(). 
*/
template <>
bool FFTOp<float, CPUContext>::RunOnDevice() {
  // input check
  auto& XReal = Input(0);
  auto& XImag = Input(1);
  auto* YReal = Output(0);
  auto* YImag = Output(1);
  CAFFE_ENFORCE_EQ(XReal.ndim(), 2);
  CAFFE_ENFORCE_EQ(XReal.ndim(), XImag.ndim());

  // output check 
  auto dims = XReal.dims();
  int N = dims[0], D = dims[1];
  YReal->ResizeLike(XReal);
  YImag->ResizeLike(XReal);

  // computing
  using namespace Eigen;
  typedef Matrix<float, -1, -1, RowMajor> RowMajorMatrixXf;
  typedef const Matrix<float, -1, -1, RowMajor> ConstRowMajorMatrixXf;
  typedef Matrix<std::complex<float>, -1, -1, RowMajor> RowMajorMatrixXcf;
  FFT<float> fft;

  RowMajorMatrixXcf X_mat(N, D);
  X_mat.real() = Map<ConstRowMajorMatrixXf>(XReal.data<float>(), N, D);
  X_mat.imag() = Map<ConstRowMajorMatrixXf>(XImag.data<float>(), N, D);

  // batch 1D FFT forward
  RowMajorMatrixXcf Y_mat(N, D);
  for (int k = 0; k < N; k++) {
      VectorXcf tmpOut(N);
      fft.fwd(tmpOut, X_mat.row(k));
      Y_mat.row(k) = tmpOut;
  }

  // 'auto' is important here for in-place assignment!
  Map<RowMajorMatrixXf>(YReal->mutable_data<float>(), N, D) = Y_mat.real();
  Map<RowMajorMatrixXf>(YImag->mutable_data<float>(), N, D) = Y_mat.imag();

  return true;
}   //FFTOp::RunOnDevice

template <>
bool FFTGradientOp<float, CPUContext>::RunOnDevice() {
  // input check
  auto& dYReal = Input(0);
  auto& dYImag = Input(1);
  auto* dXReal = Output(0);
  auto* dXImag = Output(1);
  CAFFE_ENFORCE_EQ(dYReal.ndim(), 2);
  CAFFE_ENFORCE_EQ(dYReal.ndim(), dYImag.ndim());

  // output check
  const auto canonical_axis_input = dYReal.canonical_axis_index(1);
  int N = dYReal.size_to_dim(canonical_axis_input);
  int D = dYReal.size_from_dim(canonical_axis_input);
  dXReal->ResizeLike(dYReal);
  dXImag->ResizeLike(dYImag);

  // computing
  using namespace Eigen;
  typedef Matrix<float, -1, -1, RowMajor> RowMajorMatrixXf;
  typedef const Matrix<float, -1, -1, RowMajor> ConstRowMajorMatrixXf;
  typedef Matrix<std::complex<float>, -1, -1, RowMajor> RowMajorMatrixXcf;
  FFT<float> fft;

  RowMajorMatrixXcf dY_mat(N, D);
  dY_mat.real() = Map<ConstRowMajorMatrixXf>(dYReal.data<float>(), N, D);
  dY_mat.imag() = Map<ConstRowMajorMatrixXf>(dYImag.data<float>(), N, D);

  // batch 1D FFT forward
  RowMajorMatrixXcf dX_mat(N, D);
  for (int k = 0; k < N; k++) {
      VectorXcf tmpOut(N);
      fft.fwd(tmpOut, dY_mat.row(k));
      dX_mat.row(k) = tmpOut;
  }
  Map<RowMajorMatrixXf>(dXReal->mutable_data<float>(), N, D) = dX_mat.real();
  Map<RowMajorMatrixXf>(dXImag->mutable_data<float>(), N, D) = dX_mat.imag();

  return true;
}   //FFTGradientOp::RunOnDevice

namespace {
OpSchema::Cost CostInferenceForFFT(
     const OperatorDef& def,
     const vector<TensorShape>& in) {
   struct OpSchema::Cost cost = PointwiseCostInference<0>(def, in);
   cost.params_bytes = 0;
   return cost;
 }
} // namespace

REGISTER_CPU_OPERATOR(FFT, FFTOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(FFTGradient, FFTGradientOp<float, CPUContext>);

// Input: [XReal], output: [YReal, YImage]
OPERATOR_SCHEMA(FFT)
    .NumInputs(2)
    .NumOutputs(2)
    .CostInferenceFunction(CostInferenceForFFT)
    .SetDoc(R"DOC(
This FFT op is for complex to complex batch 1D tensor Fast Fourier Transoformation
The input is two tensors, one for the real part of the input batch data,
one for the image part of input batch data. Both of the input is batch 1D
"Tensor<T>" data.
The output is like input, one Tensor<T> for output real part, one Tensor<T>
for output image part.
For FFT algorithm, we take the Cooley-Tuky algorithm.
)DOC")
    .Input(0, "XReal", "N x D tensor for real part of input complex, \
            N denotes batch size")
    .Input(1, "XImag", "same as XReal")
    .Output(0, "YReal", "N x D tensor for real part of output complex, \
            N denotes batch size")
    .Output(1, "YImage", "same as YReal");

// Input: dYReal, dYImage, output: dXReal, dXImag
OPERATOR_SCHEMA(FFTGradient)
    .NumInputs(2)
    .NumOutputs(2)
    .SetDoc(R"DOC(
FFTGradient takes both dYReal and dYImag and uses this to update dXReal 
and dXImag(actually not updated, just for computation convenience) according 
to the chain rule
)DOC");

class GetFFTGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
      def_.type() + "Gradient",
      "",
      vector<string>{GO(0), GO(1)},
      vector<string>{GI(0), GI(1)});
  }
};
REGISTER_GRADIENT(FFT, GetFFTGradient);
}  // namespace caffe2
