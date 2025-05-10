#ifndef OPENMP_REDUCTION_H
#define OPENMP_REDUCTION_H

#include <omp.h>
#include "Matrix.h"
#include "FlatMatrix.h"

Matrix multiply_openmp_reduction(const Matrix& A, const Matrix& B) {
    int N = A.rows;
    Matrix C(N, B.cols, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (int k = 0; k < A.cols; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

FlatMatrix multiply_flat_openmp_reduction(const FlatMatrix& A, const FlatMatrix& B) {
    int N = A.rows;
    FlatMatrix C(N, B.cols, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (int k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    return C;
}

FlatMatrix multiply_flat_transposed_openmp_reduction(FlatMatrix& A, FlatMatrix& B) {
    int N = A.rows;
    FlatMatrix B_T = B;
    B_T.transpose();
    FlatMatrix C(N, B_T.rows, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < B_T.rows; ++j) {
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (int k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B_T(j, k);
            }
            C(i, j) = sum;
        }
    }

    return C;
}

#endif
