#ifndef CUDA_MUL_H
#define CUDA_MUL_H

#include "Matrix.h"
#include "FlatMatrix.h"
Matrix multiply_cuda(const Matrix& A, const Matrix& B);
FlatMatrix multiply_flat_cuda(const FlatMatrix& A, const FlatMatrix& B);
FlatMatrix multiply_flat_transposed_cuda(FlatMatrix& A, FlatMatrix& B);

#endif