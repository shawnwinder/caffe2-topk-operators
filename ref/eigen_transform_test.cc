#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <typeinfo>
#include <complex>

#include <Eigen/Core>
#include <unsupported/Eigen/FFT>

using namespace std;
using namespace Eigen;

typedef Matrix<float, -1, -1, RowMajor> RowMajorMatrixXf;
typedef const Matrix<float, -1, -1, RowMajor> ConstRowMajorMatrixXf;
typedef Matrix<complex<float>, -1, -1, RowMajor> RowMajorMatrixXcf;
typedef const Matrix<complex<float>, -1, -1, RowMajor> ConstRowMajorMatrixXcf;



vector<float> get_1D_std_vector_from_matrix(MatrixXf mat) {
    // assume mat is not empty!
    MatrixXf tmp = mat.transpose();

    // initailzed from range iterators of std::vector<T>
    // e.g. vector<T>(vec.begin(), vec.begin() + vec.size()) 
    vector<float> vec(tmp.data(), tmp.data() + tmp.size());
    return vec;
}

vector<vector<float>> get_2D_std_vector_from_matrix(MatrixXf m) {
    // assume mat is not empty!
    // Eigen::MatrixXf tmp = mat.transpose();
    Eigen::MatrixXf mat = m.transpose();

    int row_length = mat.rows();
    int col_length = mat.cols();
    vector<vector<float>> vec_2d;
    for (int i = 0; i < col_length; ++ i) {
        const float* col_begin = &mat.col(i).data()[0];
        vec_2d.push_back(vector<float>(col_begin, col_begin + row_length));
    }

    return vec_2d;
}

MatrixXf get_matrix_from_2D_std_vector(vector<vector<float> > vec) {
    // assume vec is not empty!
    int row_length = vec.size();
    int col_length = vec[0].size();
    MatrixXf mat(row_length, col_length);
    for (int i = 0; i < row_length; ++ i) {
        mat.row(i) = VectorXf::Map(&vec[i][0], vec[i].size());
    }
    return mat;
}

MatrixXf get_matrix_from_1D_std_vector(vector<float> vec) {
    // assume vec is not empty!
    int length = vec.size();
    int row_length = int(sqrt(length));
    int col_length = length / row_length;
    auto mat = Map<MatrixXf>(&vec[0], row_length, col_length);
    return mat;
    
    
    // MatrixXf in;
    // in << 1, 2, 3;
    // return in;
}


void complex_matrix_testing() {
    // get real & image part from a complex matrix
    MatrixXcf mat = MatrixXcf::Random(3,3);
    MatrixXf r_mat = mat.real();
    MatrixXf i_mat = mat.imag();

    // validate the address of real * image part [different address!]
    cout << "===================================================" << endl;
    cout << "complex matrix: \n" << mat << endl;
    cout << "real part of complex matrix:\n" << r_mat << endl;
    cout << "image part of complex matrix:\n" << i_mat << endl;

    cout << "===================================================" << endl;
    cout << "address of real part:\n" << &r_mat.row(0).data()[0] << endl;
    cout << "address of real part2:\n" << &r_mat.col(0).data()[0] << endl;
    cout << "address of image part:\n" << &i_mat.row(0).data()[0] << endl;
    cout << "address of image part2:\n" << &i_mat.col(0).data()[0] << endl;

    cout << "===================================================" << endl;
    cout << "complex mat address:\n" << &mat.row(0).data()[0] << endl;
    cout << "complex mat real part address:\n" << &mat.real().data()[0] << endl;
    cout << "complex mat imag part address:\n" << &mat.imag().data()[0] << endl;
}

void complex_matrix_testing2() {
    MatrixXf r_mat;
    MatrixXf i_mat;
    r_mat.setOnes(3,3);
    i_mat = 2 * i_mat.setOnes(3,3);
    cout << "r_mat:\n" << r_mat << endl;
    cout << "i_mat:\n" << i_mat << endl;

    MatrixXcf mat(3,3);
    mat.real() = r_mat;
    mat.imag() = i_mat;
    cout << "complex mat:\n" << mat << endl;

    cout << "real mat address:\n" << &r_mat.row(0).data()[0] << endl;
    cout << "image mat address:\n" << &i_mat.row(0).data()[0] << endl;
    cout << "complex mat address:\n" << &mat.row(0).data()[0] << endl;
}

// Debug for real-to-complex RFFT (RFFT) test
void RFFT_example() {
    Eigen::FFT<float> FFT;
    Eigen::MatrixXf in(3, 4);
    in << 1,2,3,4,5,6,7,8,9,10,11,12;

    cout << "Input real matrix: \n" << in << endl;

    Eigen::MatrixXcf out;
    int dim_x = 3;
    int dim_y = 4;

    out.setZero(dim_x, dim_y);

    // apply row RFFT only equals batch 1D RFFT
    for (int k = 0; k < in.rows(); k++) {
        Eigen::VectorXcf tmpOut(dim_x);  // dim_x not dim_y!
        FFT.fwd(tmpOut, in.row(k));
        out.row(k) = tmpOut;
    }

    // apply col RFFT equals 2D RFFT
    /*
    for (int k = 0; k < in.cols(); k++) {
        Eigen::VectorXcf tmpOut(dim_y);
        FFT.fwd(tmpOut, out.col(k));
        out.col(k) = tmpOut;
    }
    */

    cout << "Output complex matrix: \n" << out << endl;
    cout << "Input real matrix: \n" << in << endl;
}

// Debug for in-place Matrix assignment
void RFFT_example2() {
    FFT<float> FFT;
    int dim_x = 3;
    int dim_y = 4;

    const vector<float> vec = {1,2,3,4,5,6,7,8,9,10,11,12};
    MatrixXf in = Map<const Matrix<float, Dynamic, Dynamic, RowMajor>>(&vec[0], dim_x, dim_y);
    // MatrixXf in = Map<const MatrixXf, RowMajor>(&vec[0], dim_x, dim_y);  // not working!
    cout << "Input real matrix: \n" << in << endl;

    // apply row RFFT only equals batch 1D RFFT
    // MatrixXcf out(dim_x, dim_y);
    MatrixXcf out = Matrix<std::complex<float>, -1, -1, RowMajor>(dim_x, dim_y);
    for (int k = 0; k < in.rows(); k++) {
        Eigen::VectorXcf tmpOut(dim_x);  // dim_x not dim_y!
        FFT.fwd(tmpOut, in.row(k));
        out.row(k) = tmpOut;
    }

    // MatrixXf real = out.real();
    Matrix<float, -1, -1, RowMajor> real = out.real();
    auto real_begin = &real.data()[0];
    vector<float> raw_real(real_begin, real_begin + dim_x * dim_y);

    // MatrixXf imag = out.imag();
    Matrix<float, -1, -1, RowMajor> imag = out.imag();
    auto imag_begin = &imag.data()[0];
    vector<float> raw_imag(imag_begin, imag_begin + dim_x * dim_y);


    cout << "Output complex matrix: \n" << out << endl;
    cout << "Output complex matrix real part: \n" << real << endl;
    cout << "Output complex matrix imag part: \n" << imag << endl;

    cout << "raw real:\n";
    for (auto n : raw_real) cout << n << ",";
    cout << endl;
    cout << "raw imag:\n";
    for (auto n : raw_imag) cout << n << ",";
    cout << endl;
}

// Debug for "ConstRowMajorMatrixXf" as lvalue
void RFFT_example3() {
    FFT<float> FFT;
    int dim_x = 3;
    int dim_y = 4;

    const vector<float> vec = {1,2,3,4,5,6,7,8,9,10,11,12};
    typedef Matrix<float, -1, -1, RowMajor> RowMajorMatrixXf;
    typedef const Matrix<float, -1, -1, RowMajor> ConstRowMajorMatrixXf;

    // ConstRowMajorMatrixXf in = Map<ConstRowMajorMatrixXf>(&vec[0], dim_x, dim_y);
    RowMajorMatrixXf in = Map<ConstRowMajorMatrixXf>(&vec[0], dim_x, dim_y);
    // MatrixXf in = Map<const MatrixXf>(&vec[0], dim_x, dim_y);
    cout << "input mat:\n" << in << endl;
    
    // apply row RFFT only for batch 1D RFFT
    MatrixXcf out(dim_x, dim_y);
    for (int k = 0; k < dim_x; k++) {
        VectorXcf tmpOut(dim_x);
        FFT.fwd(tmpOut, in.row(k));
        out.row(k) = tmpOut;
    }
    cout << "out mat:\n" << out << endl;
}

void RFFT_example4() {
    FFT<float> FFT;
    int dim_x = 3;
    int dim_y = 4;

    const vector<float> vec = {1,2,3,4,5,6,7,8,9,10,11,12};
    typedef Matrix<float, -1, -1, RowMajor> RowMajorMatrixXf;
    typedef const Matrix<float, -1, -1, RowMajor> ConstRowMajorMatrixXf;
    typedef Matrix<complex<float>, -1, -1, RowMajor> RowMajorMatrixXcf;

    // ConstRowMajorMatrixXf in = Map<ConstRowMajorMatrixXf>(&vec[0], dim_x, dim_y);
    RowMajorMatrixXf in = Map<ConstRowMajorMatrixXf>(&vec[0], dim_x, dim_y);
    cout << "input mat:\n" << in << endl;
    
    // apply row RFFT only for batch 1D RFFT
    MatrixXcf out(dim_x, dim_y);
    // MatrixXcf out(dim_x, (dim_y / 2) + 1);
    for (int k = 0; k < dim_x; k++) {
        VectorXcf tmpOut(dim_x);
        FFT.fwd(tmpOut, in.row(k));
        out.row(k) = tmpOut;
    }

    // auto real_out = Map<RowMajorMatrixXcf, 0, OuterStride<>>(
    auto real_out = Map<MatrixXcf, 0, OuterStride<>>(
            out.data(),
            dim_x,
            dim_y / 2 + 1,
            OuterStride<>(out.outerStride()));

    // cout << "out mat:\n" << out << endl;
    cout << "real out mat:\n" << real_out << endl;
    cout << "real part of real_out:\n" << real_out.real() << endl;
    cout << "imag part of real_out:\n" << real_out.imag() << endl;
}

void IRFFT_example() {
    int dim_x = 3;
    int dim_y = 4;
    Eigen::FFT<float> FFT;
    Eigen::MatrixXf in(dim_x, dim_y);
    in << 1,2,3,4,5,6,7,8,9,10,11,12;
    // in << 1,2,3,4,5,6,7,8,9;
    // MatrixXf in = MatrixXf::Random(dim_x, dim_y);
    cout << "Input real matrix: \n" << in << endl;

    Eigen::MatrixXcf out(dim_x, dim_y);
    // apply row RFFT 
    for (int k = 0; k < in.rows(); k++) {
        Eigen::VectorXcf tmpOut(dim_x);  // dim_x not dim_y!
        FFT.fwd(tmpOut, in.row(k));
        out.row(k) = tmpOut;
    }
    cout << "Output complex matrix: \n" << out << endl;

    // apply row iRFFT 
    MatrixXf recover(dim_x, dim_y);
    for (int k = 0; k < out.rows(); k++) {
        Eigen::VectorXf tmpOut(dim_x);  // dim_x not dim_y!
        FFT.inv(tmpOut, out.row(k));
        recover.row(k) = tmpOut;
    }
    cout << "Recovered real matrix: \n" << recover << endl;
}

// test manual assigned value for complex-to-real RFFT (iRFFT)
void IRFFT_example2() {
    // vector<complex> initailization 1
    vector<float> real = {10, -2, -2, -2, 26, -2, -2, -2, 42, -2, -2, -2};
    vector<float> imag = {0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2};
    vector<complex<float>> c_vec(real.size());
    std::transform(
            real.begin(), 
            real.end(), 
            imag.begin(), 
            c_vec.begin(),
            [](float fa, float fb) {return complex<float>(fa, fb);}
            );
    
    // vector<complex> initailization 2
    // vector<complex<float>> c_vec = {
    //     complex<float>(10,0), complex<float>(-2,2), complex<float>(-2,0), complex<float>(-2,-2),
    //     complex<float>(26,0), complex<float>(-2,2), complex<float>(-2,0), complex<float>(-2,-2),
    //     complex<float>(42,0), complex<float>(-2,2), complex<float>(-2,0), complex<float>(-2,-2)
    // };

    int dim_x = 3;
    int dim_y = 4;
    Eigen::FFT<float> FFT;
    auto cmat = Map<Matrix<complex<float>, -1, -1, RowMajor>>(&c_vec[0], dim_x, dim_y);
    cmat /= 2;
    cout << "Original complex matrix:\n" << cmat << endl;

    // apply row iRFFT 
    Matrix<float, -1, -1, RowMajor> recover(dim_x, dim_y); // real matrix is enough!
    for (int k = 0; k < dim_x; k++) {
        Eigen::VectorXf tmpOut(dim_x);  // dim_x not dim_y!
        FFT.inv(tmpOut, cmat.row(k));
        recover.row(k) = tmpOut;
    }
    cout << "Recovered real matrix:\n" << recover << endl;
}

// halp spectrum of RFFT result test
void IRFFT_example3() {
    int dim_x = 3;
    int dim_y = 4;
    Eigen::FFT<float> FFT;
    Eigen::MatrixXf in(dim_x, dim_y);
    in << 1,2,3,4,5,6,7,8,9,10,11,12;
    cout << "Input real matrix: \n" << in << endl;

    Eigen::MatrixXcf out_tmp(dim_x, dim_y);
    // apply row RFFT 
    for (int k = 0; k < in.rows(); k++) {
        Eigen::VectorXcf tmpOut(dim_x);  // dim_x not dim_y!
        FFT.fwd(tmpOut, in.row(k));
        out_tmp.row(k) = tmpOut;
    }
    auto out = Map<MatrixXcf, 0, OuterStride<>>(
            out_tmp.data(),
            dim_x,
            dim_y / 2 + 1,
            OuterStride<>(out_tmp.outerStride()));
    cout << "Output complex matrix: \n" << out << endl;

    
    // add the other random half spectrum of out
    MatrixXcf half = MatrixXcf::Random(dim_x, dim_y / 2 - 1);
    MatrixXcf whole(dim_x, dim_y);
    whole << out, half;

    cout << "The whole part of the RFFT forward result:\n" << whole << endl;


    // apply row iRFFT 
    MatrixXf recover(dim_x, dim_y);
    for (int k = 0; k < whole.rows(); k++) {
        Eigen::VectorXf tmpOut(dim_x);  // dim_x not dim_y!
        FFT.inv(tmpOut, whole.row(k));
        recover.row(k) = tmpOut;
    }
    cout << "Recovered real matrix: \n" << recover << endl;
}

// debug for inplace map
void Map_test() {
    vector<float> vec = {1,2,3,4,5,6,7,8,9,10,11,12};
    
    // in-place
    // auto mat = Map<Matrix<float, -1, -1, RowMajor>>(&vec[0], 3, 4);  
    // non-inplace
    MatrixXf mat = Map<Matrix<float, -1, -1, RowMajor>>(&vec[0], 3, 4);
    string type_ = typeid(mat).name();
    cout << "mat type: " << type_ << endl;
    
    mat = 3 * mat;
    cout << "mat:\n" << mat << endl;

    cout << "vec:\n";
    for (auto n:vec) cout << n << ",";
    cout << endl;
}

void matrix_slicing_test() {
    typedef Matrix<float, -1, -1, RowMajor> RowMajorMatrixXf;
    int N = 3, D = 4;
    RowMajorMatrixXf mat(N, D);
    mat.setOnes(N, D);
    // mat << 1,2,3,4,5,6,7,8,9,10,11,12;
    cout << "==============================================" << endl;
    cout << "original mat:\n" << mat << endl;

    /* mat[..., 1:] (numpy stype slicing) */
    /// method1: using Map with OuterStride
    auto v1 = Map<RowMajorMatrixXf, 0, OuterStride<>>(
            mat.col(1).data(), 
            N, 
            D - 1,
            OuterStride<>(mat.outerStride()));
    v1 /= 2;
    cout << "==============================================" << endl;
    cout << "slicing view:\n" << v1 << endl;
    cout << "original mat:\n" << mat << endl;

    // method2: using Matrix::block() (this way is not flexible)
    // // auto b1 = mat.block<N, D - 1>(0, 1);  // not allowed! N or D must be constexpr!
    // auto b1 = mat.block<3, 3>(0, 1);
    // b1 /= 2;
    // cout << "==============================================" << endl;
    // cout << "mat block:\n" << b1 << endl;
    // cout << "original mat:\n" << mat << endl;
    

    /* mat[..., 1:-1] (numpy stype slicing) */
    // auto v2 = Map<RowMajorMatrixXf, 0, OuterStride<>>(
    //         mat.col(1).data(), 
    //         N, 
    //         D - 2,
    //         OuterStride<>(mat.outerStride()));
    // v2 /= 2;
    // cout << "==============================================" << endl;
    // cout << "slicing view:\n" << v2 << endl;
    // cout << "original mat:\n" << mat << endl;
}

// C2C forward fft test
void FFT_example() {
    FFT<float> FFT;
    int dim_x = 3;
    int dim_y = 4;

    const vector<float> v_real = {1,2,3,4,5,6,7,8,9,10,11,12};
    const vector<float> v_imag(dim_x * dim_y, 0);
    RowMajorMatrixXcf in(dim_x, dim_y);
    in.real() = Map<ConstRowMajorMatrixXf>(&v_real[0], dim_x, dim_y);
    in.imag() = Map<ConstRowMajorMatrixXf>(&v_imag[0], dim_x, dim_y);
    cout << "Input real matrix: \n" << in << endl;

    // apply batch 1D FFT
    RowMajorMatrixXcf out(dim_x, dim_y);
    for (int k = 0; k < dim_x; k++) {
        VectorXcf tmpOut(dim_x);  // dim_x not dim_y!
        FFT.fwd(tmpOut, in.row(k));
        out.row(k) = tmpOut;
    }

    cout << "Output complex matrix: \n" << out << endl;
    cout << "Output complex matrix real part: \n" << out.real() << endl;
    cout << "Output complex matrix imag part: \n" << out.imag() << endl;
}

// C2C inverse fft test
void IFFT_example() {
    FFT<float> FFT;
    int dim_x = 3;
    int dim_y = 4;
    const vector<float> v_real = {1,2,3,4,5,6,7,8,9,10,11,12};
    const vector<float> v_imag(dim_x * dim_y, 0);
    RowMajorMatrixXcf in(dim_x, dim_y);
    in.real() = Map<ConstRowMajorMatrixXf>(&v_real[0], dim_x, dim_y);
    in.imag() = Map<ConstRowMajorMatrixXf>(&v_imag[0], dim_x, dim_y);
    cout << "Input real matrix: \n" << in << endl;
    cout << endl;

    // FFT fwd
    RowMajorMatrixXcf out(dim_x, dim_y);
    for (int k = 0; k < dim_x; k++) {
        VectorXcf tmpOut(dim_x);  // dim_x not dim_y!
        FFT.fwd(tmpOut, in.row(k));
        out.row(k) = tmpOut;
    }

    cout << "Output complex matrix: \n" << out << endl;
    cout << "Output complex matrix real part: \n" << out.real() << endl;
    cout << "Output complex matrix imag part: \n" << out.imag() << endl;
    cout << endl;

    // FFT inv
    RowMajorMatrixXcf recover(dim_x, dim_y);
    for (int k = 0; k < dim_x; k++) {
        VectorXcf tmpOut(dim_x);  // dim_x not dim_y!
        FFT.inv(tmpOut, out.row(k));
        recover.row(k) = tmpOut;
    }
    
    cout << "Recovered complex matrix: \n" << recover << endl;
    cout << "Recovered complex matrix real part: \n" << recover.real() << endl;
    cout << "Recovered complex matrix imag part: \n" << recover.imag() << endl;
    cout << endl;
}

// reshape test for eigen matrix
void reshape_test() {
    int dim_x = 3;
    int dim_y = 4;
    // const vector<float> v_real = {1,2,3,4,5,6,7,8,9,10,11,12};
    // auto mat = Ma

    RowMajorMatrixXf mat(dim_x, dim_y);
    mat << 1,2,3,4,5,6,7,8,9,10,11,12;

    cout << "original mat:\n" << mat << endl;

    // Map<RowMajorMatrixXf> mat_reshaped(mat.data(), dim_y, dim_x);
    // Map<MatrixXf> mat_reshaped(mat.data(), dim_y, dim_x);
    // Map<MatrixXf> mat_reshaped(mat.data(), dim_y, -1);
    Map<MatrixXf> mat_reshaped(mat.data(), 2, 2, 3);

    cout << "reshaped mat:\n" << mat_reshaped << endl;


}

int main () {
    /// test_1
    // Eigen::Matrix3f mat;
    // mat << 1,2,3,4,5,6,7,8,9;
    // vector<float> vec = get_1D_std_vector_from_matrix(mat);
    // for (auto n: vec) cout << n << ", ";
    // cout << endl;

    /// test_2
    // vector<vector<float> > vec = {{1,2,3}, {4,5,6}, {7,8,9}};
    // MatrixXf mat = get_matrix_from_2D_std_vector(vec);
    // cout << mat << endl;
    
    /// test_3
    // RFFT_example();
    
    /// test_4
    // Eigen::Matrix3f mat;
    // mat << 1,2,3,4,5,6,7,8,9;
    // vector<vector<float>> vec_2d = get_2D_std_vector_from_matrix(mat);
    // for (auto row : vec_2d) {
    //     for (auto n : row) cout << n << ", ";
    //     cout << endl;
    // }

    /// test_5
    // vector<float> v = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
    // auto mat = get_matrix_from_1D_std_vector(v);
    // cout << mat << endl;

    /// test_6
    // complex_matrix_testing();
    
    /// test_7
    // complex_matrix_testing2();
    
    /// test_8
    // IRFFT_example();

    /// test_9
    // RFFT_example2();
    
    /// test_10
    // Map_test();

    /// test_11
    // RFFT_example3();
    
    /// test_12
    // IRFFT_example2();
    
    /// test_13
    // matrix_slicing_test();
    
    /// test_14
    // RFFT_example4();

    /// test_15
    // IRFFT_example3();

    /// test_16
    // FFT_example();

    /// test_17
    // IFFT_example();
    
    /// reshape test
    reshape_test();
    
    return 0;
}

