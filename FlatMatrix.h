#ifndef FLATMATRIX_H
#define FLATMATRIX_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <fstream>
#include <string>

using namespace std;

class FlatMatrix {
public:
    int rows, cols;
    vector<double> data;

    FlatMatrix() : rows(0), cols(0), data() {}

    FlatMatrix(int rows, int cols, double init_val = 0.0)
        : rows(rows), cols(cols), data(rows * cols, init_val) {}

    FlatMatrix(int rows, int cols, const vector<double>& init_data)
        : rows(rows), cols(cols), data(init_data) {
        if (init_data.size() != static_cast<size_t>(rows * cols)) {
            throw invalid_argument("Data size does not match matrix dimensions.");
        }
    }

    double& operator()(int i, int j) {
        if (i < 0 || i >= rows || j < 0 || j >= cols) {
            throw out_of_range("Index out of range");
        }
        return data[i * cols + j];
    }

    const double& operator()(int i, int j) const {
        if (i < 0 || i >= rows || j < 0 || j >= cols) {
            throw out_of_range("Index out of range");
        }
        return data[i * cols + j];
    }

    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cout << setw(10) << (*this)(i, j) << " ";
            }
            cout << endl;
        }
    }

    static FlatMatrix multiply(const FlatMatrix& A, const FlatMatrix& B) {
        if (A.cols != B.rows) {
            throw invalid_argument("Matrix dimensions do not match for multiplication");
        }

        FlatMatrix C(A.rows, B.cols, 0.0);

        for (int i = 0; i < A.rows; ++i) {
            for (int j = 0; j < B.cols; ++j) {
                for (int k = 0; k < A.cols; ++k) {
                    C(i, j) += A(i, k) * B(k, j);
                }
            }
        }

        return C;
    }

    void transpose() {
        vector<double> new_data(rows * cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                new_data[j * rows + i] = data[i * cols + j];
            }
        }

        swap(rows, cols);
        data = move(new_data);
    }

    double* getData() {
        return data.data();
    }

    const double* getData() const {
        return data.data();
    }

    FlatMatrix multiply_flat_transposed(FlatMatrix& A, FlatMatrix& B) {
        if (A.cols != B.rows) {
            throw invalid_argument("Matrix dimensions do not match for multiplication");
        }
    
        B.transpose();
    
        FlatMatrix C(A.rows, B.cols, 0.0);
    
        for (int i = 0; i < A.rows; ++i) {
            for (int j = 0; j < B.cols; ++j) {
                for (int k = 0; k < A.cols; ++k) {
                    C(i, j) += A(i, k) * B(k, j);
                }
            }
        }
    
        return C;
    }
};

inline bool loadFlatMatrix(const string& filename, FlatMatrix& mat) {
    ifstream in(filename);
    if (!in.is_open()) return false;

    string line;
    vector<vector<double>> temp;
    while (getline(in, line)) {
        vector<double> row;
        size_t start = 0;
        while ((start = line.find_first_of("0123456789.-", start)) != string::npos) {
            size_t end = line.find_first_not_of("0123456789.eE-+", start);
            row.push_back(stod(line.substr(start, end - start)));
            start = end;
        }
        if (!row.empty()) temp.push_back(row);
    }

    if (temp.empty()) return false;

    size_t cols = temp[0].size();
    for (const auto& row : temp) {
        if (row.size() != cols) {
            cerr << "Inconsistent number of columns in file: " << filename << endl;
            return false;
        }
    }

    mat.rows = temp.size();
    mat.cols = cols;
    mat.data.clear();
    for (const auto& row : temp) {
        mat.data.insert(mat.data.end(), row.begin(), row.end());
    }

    return true;
}

inline bool compareFlatMatrices(const FlatMatrix& A, const FlatMatrix& B, double epsilon = 1e-4) {
    if (A.rows != B.rows || A.cols != B.cols) return false;
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            if (abs(A(i, j) - B(i, j)) > epsilon) return false;
        }
    }
    return true;
}

#endif
