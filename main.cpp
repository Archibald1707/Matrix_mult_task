#include <iostream>
#include <chrono>
#include "Matrix.h"
#include "thread_atomic.h"
#include "openmp_reduction.h"
#include "cuda_mul.h"

using namespace std;

void printUsage(const char* name) {
    cout << "Usage: " << name << " <mode> [size]" << endl;
    cout << "Modes:" << endl;
    cout << "  atomic          - std::thread + atomic (Matrix)" << endl;
    cout << "  openmp          - OpenMP reduction (Matrix)" << endl;
    cout << "  cuda            - CUDA (Matrix)" << endl;
    cout << "  atomic_flat     - std::thread + atomic (FlatMatrix)" << endl;
    cout << "  openmp_flat     - OpenMP reduction (FlatMatrix)" << endl;
    cout << "  cuda_flat       - CUDA (FlatMatrix)" << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        // test mode
        Matrix A, B;
        if (!loadMatrix("matrix.in.txt", A)) {
            cerr << "Failed to load matrix.in.txt" << endl;
            return 1;
        }
        B = A;

        Matrix C, ref;
        if (!loadMatrix("matrix.out.txt", ref)) {
            cerr << "Failed to load matrix.out.txt" << endl;
            return 1;
        }

        C = multiply_thread_atomic(A, B);
        bool ok_atomic = compareMatrices(C, ref);

        C = multiply_openmp_reduction(A, B);
        bool ok_openmp = compareMatrices(C, ref);

        C = multiply_cuda(A, B);
        bool ok_cuda = compareMatrices(C, ref);

        cout << "Test mode results:\n";
        cout << "  atomic: " << (ok_atomic ? "OK" : "FAILED") << endl;
        cout << "  openmp: " << (ok_openmp ? "OK" : "FAILED") << endl;
        cout << "  cuda:   " << (ok_cuda   ? "OK" : "FAILED") << endl;

        FlatMatrix flatA, flatB, flatC, flatRef;

        if (!loadFlatMatrix("matrix.in.txt", flatA)) {
            cerr << "Failed to load matrix.in.txt as FlatMatrix" << endl;
            return 1;
        }
        flatB = flatA;
        
        if (!loadFlatMatrix("matrix.out.txt", flatRef)) {
            cerr << "Failed to load matrix.out.txt as FlatMatrix" << endl;
            return 1;
        }        

        flatC = multiply_flat_thread_atomic(flatA, flatB);
        bool ok_flat_atomic = compareFlatMatrices(flatC, flatRef);

        flatC = multiply_flat_openmp_reduction(flatA, flatB);
        bool ok_flat_openmp = compareFlatMatrices(flatC, flatRef);

        flatC = multiply_flat_cuda(flatA, flatB);
        bool ok_flat_cuda = compareFlatMatrices(flatC, flatRef);

        cout << "FlatMatrix test mode results:\n";
        cout << "  atomic_flat: " << (ok_flat_atomic ? "OK" : "FAILED") << endl;
        cout << "  openmp_flat: " << (ok_flat_openmp ? "OK" : "FAILED") << endl;
        cout << "  cuda_flat:   " << (ok_flat_cuda   ? "OK" : "FAILED") << endl;

        flatC = multiply_flat_transposed_thread_atomic(flatA, flatB);
        bool ok_flat_transposed_atomic = compareFlatMatrices(flatC, flatRef);

        flatB = flatA;

        flatC = multiply_flat_transposed_openmp_reduction(flatA, flatB);
        bool ok_flat_transposed_openmp = compareFlatMatrices(flatC, flatRef);

        flatB = flatA;
        flatB.transpose();
        
        flatC = multiply_flat_transposed_cuda(flatA, flatB);
        bool ok_flat_transposed_cuda = compareFlatMatrices(flatC, flatRef);

        flatB = flatA;

        cout << "FlatMatrix with transposed B test mode results:\n";
        cout << "  atomic_flat_transposed: " << (ok_flat_transposed_atomic ? "OK" : "FAILED") << endl;
        cout << "  openmp_flat_transposed: " << (ok_flat_transposed_openmp ? "OK" : "FAILED") << endl;
        cout << "  cuda_flat_transposed:   " << (ok_flat_transposed_cuda   ? "OK" : "FAILED") << endl;

        return 0;
    }

    string mode = argv[1];
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    int N = stoi(argv[2]);
    Matrix A(N, N), B(N, N), C;
    FlatMatrix FA(N, N), FB(N, N), FC;

    srand(42);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double val1 = static_cast<double>(rand()) / RAND_MAX;
            double val2 = static_cast<double>(rand()) / RAND_MAX;
            A[i][j] = val1;
            B[i][j] = val2;
            FA(i, j) = val1;
            FB(i, j) = val2;
        }
    
    auto start = chrono::high_resolution_clock::now();

    if (mode == "atomic") {
        C = multiply_thread_atomic(A, B);
    } else if (mode == "openmp") {
        C = multiply_openmp_reduction(A, B);
    } else if (mode == "cuda") {
        C = multiply_cuda(A, B);
    } else if (mode == "atomic_flat") {
        FC = multiply_flat_thread_atomic(FA, FB);
    } else if (mode == "openmp_flat") {
        FC = multiply_flat_openmp_reduction(FA, FB);
    } else if (mode == "cuda_flat") {
        FC = multiply_flat_cuda(FA, FB);
    } else if (mode == "atomic_flat_transposed") {
        FC = multiply_flat_transposed_thread_atomic(FA, FB);
    } else if (mode == "openmp_flat_transposed") {
        FC = multiply_flat_transposed_openmp_reduction(FA, FB);
    } else if (mode == "cuda_flat_transposed") {
        FC = multiply_flat_transposed_cuda(FA, FB);
    } else {
        printUsage(argv[0]);
        return 1;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Execution time (" << mode << "): " << elapsed.count() << " seconds" << endl;
    return 0;
}
