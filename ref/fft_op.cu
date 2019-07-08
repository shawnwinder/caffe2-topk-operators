#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/fft_op.h"

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include "cufft.h"


namespace caffe2 {
namespace {
__global__ void GetComplexKernel(const int N, const cufftReal* Real_, 
        const cufftReal* Imag_, cufftComplex* Complex_) {
    CUDA_1D_KERNEL_LOOP(i, N) {
        Complex_[i].x = Real_[i]; 
        Complex_[i].y = Imag_[i]; 
    }
}

__global__ void GetRealAndImageKernel(const int N, const cufftComplex* Complex_, 
        cufftReal* Real_, cufftReal* Imag_) {
    CUDA_1D_KERNEL_LOOP(i, N) {
        Real_[i] = Complex_[i].x;
        Imag_[i] = Complex_[i].y;
    }
}
}  // namespace

template <>
bool FFTOp<float, CUDAContext>::RunOnDevice() {
  /// schema checking
  auto& XReal = Input(0);
  auto& XImag = Input(1);
  auto* YReal = Output(0);
  auto* YImag = Output(1);
  CAFFE_ENFORCE_EQ(XReal.ndim(), 2);
  CAFFE_ENFORCE_EQ(XReal.ndim(), XImag.ndim());

  const auto canonical_axis_input = XReal.canonical_axis_index(1);
  int N = XReal.size_to_dim(canonical_axis_input);
  int D = XReal.size_from_dim(canonical_axis_input);
  YReal->ResizeLike(XReal);
  YImag->ResizeLike(XImag);

  /// computing
  // memory allocation
  cufftComplex *XComplex; 
  CUDA_ENFORCE(cudaMalloc(&XComplex, N * D *  sizeof(cufftComplex)));
  cufftComplex *YComplex; 
  CUDA_ENFORCE(cudaMalloc(&YComplex, N * D *  sizeof(cufftComplex)));

  // merge input real and imag part for cufftExecC2C transformation
  GetComplexKernel<<<
      CAFFE_GET_BLOCKS(XReal.size()),
      CAFFE_CUDA_NUM_THREADS, 
      0, context_.cuda_stream()>>>( 
          XReal.size(),
          XReal.data<float>(),
          XImag.data<float>(),
          XComplex);

  // batched 1D FFT forward
  cufftHandle handle;
  int rank = 1;                       // 1D FFTs
  int n[] = {D};                      // Size of the Fourier transform
  int istride = 1, ostride = 1;       // Distance between two successive input/output elements
  int idist = D, odist = D;           // Distance between batches
  int inembed[] = { 0 };              // Input size with pitch (ignored for 1D transforms)
  int onembed[] = { 0 };              // Output size with pitch (ignored for 1D transforms)
  int batch = N;                      // Number of batched executions
  cufftPlanMany( 
      &handle, rank, n, 
      inembed, istride, idist, 
      onembed, ostride, odist, 
      CUFFT_C2C, 
      batch);
  cufftExecC2C(handle, XComplex, YComplex, CUFFT_FORWARD);
  cufftDestroy(handle);

  // split real and image part of FFT output
  GetRealAndImageKernel<<<
      CAFFE_GET_BLOCKS(YReal->size()), 
      CAFFE_CUDA_NUM_THREADS, 
      0, context_.cuda_stream()>>>( 
              YReal->size(),
              YComplex,
              YReal->mutable_data<float>(), 
              YImag->mutable_data<float>());

  // free tmp GPU memory
  CUDA_ENFORCE(cudaFree(XComplex));
  CUDA_ENFORCE(cudaFree(YComplex));

  return true;
}

template <>
bool FFTGradientOp<float, CUDAContext>::RunOnDevice() {
  /// schema checking
  auto& dYReal = Input(0);
  auto& dYImag = Input(1);
  auto* dXReal = Output(0);
  auto* dXImag = Output(1);
  CAFFE_ENFORCE_EQ(dYReal.ndim(), 2);
  CAFFE_ENFORCE_EQ(dYReal.ndim(), dYImag.ndim());

  const auto canonical_axis_input = dYReal.canonical_axis_index(1);
  int N = dYReal.size_to_dim(canonical_axis_input);
  int D = dYReal.size_from_dim(canonical_axis_input);
  dXReal->ResizeLike(dYReal);
  dXImag->ResizeLike(dYImag);

  /// computing
  // memory allocation
  cufftComplex *dYComplex; 
  CUDA_ENFORCE(cudaMalloc(&dYComplex, N * D *  sizeof(cufftComplex)));
  cufftComplex *dXComplex; 
  CUDA_ENFORCE(cudaMalloc(&dXComplex, N * D *  sizeof(cufftComplex)));

  // merge input real and imag part for cufftExecC2C transformation
  GetComplexKernel<<<
      CAFFE_GET_BLOCKS(dYReal.size()),
      CAFFE_CUDA_NUM_THREADS, 
      0, context_.cuda_stream()>>>( 
          dYReal.size(),
          dYReal.data<float>(),
          dYImag.data<float>(),
          dYComplex);

  // batched 1D FFT forward
  cufftHandle handle;
  int rank = 1;                       // 1D FFTs
  int n[] = {D};                      // Size of the Fourier transform
  int istride = 1, ostride = 1;       // Distance between two successive input/output elements
  int idist = D, odist = D;           // Distance between batches
  int inembed[] = { 0 };              // Input size with pitch (ignored for 1D transforms)
  int onembed[] = { 0 };              // Output size with pitch (ignored for 1D transforms)
  int batch = N;                      // Number of batched executions
  cufftPlanMany( 
      &handle, rank, n, 
      inembed, istride, idist, 
      onembed, ostride, odist, 
      CUFFT_C2C, 
      batch);
  cufftExecC2C(handle, dYComplex, dXComplex, CUFFT_FORWARD);
  cufftDestroy(handle);

  // split real and image part of FFT output
  GetRealAndImageKernel<<<
      CAFFE_GET_BLOCKS(dXReal->size()), 
      CAFFE_CUDA_NUM_THREADS, 
      0, context_.cuda_stream()>>>( 
              dXReal->size(),
              dXComplex,
              dXReal->mutable_data<float>(), 
              dXImag->mutable_data<float>());

  // free tmp GPU memory
  CUDA_ENFORCE(cudaFree(dYComplex));
  CUDA_ENFORCE(cudaFree(dXComplex));

  return true;
}

REGISTER_CUDA_OPERATOR(FFT, FFTOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(FFTGradient, FFTGradientOp<float, CUDAContext>);
}  // namespace caffe2
