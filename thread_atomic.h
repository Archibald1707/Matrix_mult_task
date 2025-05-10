#ifndef THREAD_ATOMIC_H
#define THREAD_ATOMIC_H

#include <vector>
#include <thread>
#include <atomic>
#include "Matrix.h"
#include "FlatMatrix.h"

Matrix multiply_thread_atomic(const Matrix& A, const Matrix& B) {
    int N = A.rows;
    Matrix C(N, B.cols, 0.0);
    std::atomic<int> row(0);
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    auto worker = [&]() {
        while (true) {
            int i = row.fetch_add(1);
            if (i >= N) break;
            for (int j = 0; j < B.cols; ++j) {
                for (int k = 0; k < A.cols; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    };

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) t.join();
    return C;
}

FlatMatrix multiply_flat_thread_atomic(const FlatMatrix& A, const FlatMatrix& B) {
    int N = A.rows;
    FlatMatrix C(N, B.cols, 0.0);
    std::atomic<int> row(0);
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    auto worker = [&]() {
        while (true) {
            int i = row.fetch_add(1);
            if (i >= N) break;
            for (int j = 0; j < B.cols; ++j) {
                for (int k = 0; k < A.cols; ++k) {
                    C(i, j) += A(i, k) * B(k, j);
                }
            }
        }
    };

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) t.join();
    return C;
}

FlatMatrix multiply_flat_transposed_thread_atomic(FlatMatrix& A, FlatMatrix& B) {
    int N = A.rows;
    FlatMatrix B_T = B;
    B_T.transpose();
    FlatMatrix C(N, B_T.rows, 0.0);
    std::atomic<int> row(0);
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    auto worker = [&]() {
        while (true) {
            int i = row.fetch_add(1);
            if (i >= N) break;
            for (int j = 0; j < B_T.rows; ++j) {
                for (int k = 0; k < A.cols; ++k) {
                    C(i, j) += A(i, k) * B_T(j, k);
                }
            }
        }
    };

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) t.join();
    return C;
}

#endif
