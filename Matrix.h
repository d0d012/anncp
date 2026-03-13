#pragma once
#include <vector>
class Matrix {

public:
    Matrix(int rows, int cols);
    void setValue(int row, int col, double value);
    double getValue(int row, int col) const;
    int getRows() const;
    int getCols() const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix ReLU()const;

private:
    int rows;
    int cols;
    std::vector<std::vector<double>> data;
    
};