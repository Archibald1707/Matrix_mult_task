#ifndef MATRIX_H
#define MATRIX_H

#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <cstring>

using namespace std;

class Matrix {
public:
    int rows, cols;
    vector<vector<double>> data;

    Matrix() : rows(0), cols(0), data() {}

    Matrix(int rows, int cols, double init_val = 0.0)
        : rows(rows), cols(cols), data(rows, vector<double>(cols, init_val)) {}

    Matrix(int rows, int cols, const vector<vector<double>>& init_data)
        : rows(rows), cols(cols), data(init_data) {}

    vector<double>& operator[](int i) {
        if (i < 0 || i >= rows) {
            throw out_of_range("Index out of range");
        }
        return data[i];
    }

    const vector<double>& operator[](int i) const {
        if (i < 0 || i >= rows) {
            throw out_of_range("Index out of range");
        }
        return data[i];
    }

    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cout << setw(10) << data[i][j] << " ";
            }
            cout << endl;
        }
    }

    static Matrix multiply(const Matrix& A, const Matrix& B) {
        if (A.cols != B.rows) {
            throw invalid_argument("Matrix dimensions do not match for multiplication");
        }

        Matrix C(A.rows, B.cols, 0.0);

        for (int i = 0; i < A.rows; ++i) {
            for (int j = 0; j < B.cols; ++j) {
                for (int k = 0; k < A.cols; ++k) {
                    C[i][j] += A.data[i][k] * B.data[k][j];
                }
            }
        }

        return C;
    }
};

inline bool loadMatrix(const string& filename, Matrix& mat) {
    ifstream in(filename);
    if (!in.is_open()) return false;

    string line;
    mat.data.clear();
    while (getline(in, line)) {
        vector<double> row;
        size_t start = 0;
        while ((start = line.find_first_of("0123456789.-", start)) != string::npos) {
            size_t end = line.find_first_not_of("0123456789.eE-+", start);
            row.push_back(stod(line.substr(start, end - start)));
            start = end;
        }
        if (!row.empty()) mat.data.push_back(row);
    }
    mat.rows = mat.data.size();
    mat.cols = mat.data.empty() ? 0 : mat.data[0].size();
    return true;
}

inline bool compareMatrices(const Matrix& A, const Matrix& B, double epsilon = 1e-4) {
    if (A.rows != B.rows || A.cols != B.cols) return false;
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            if (abs(A[i][j] - B[i][j]) > epsilon) return false;
        }
    }
    return true;
}

#endif
