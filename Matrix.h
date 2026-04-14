#pragma once
#include <vector>
class Matrix {

public:
  Matrix(int rows, int cols);
  Matrix() : rows(0), cols(0) {}

  void setValue(int row, int col, double value);
  double getValue(int row, int col) const;
  int getRows() const;
  int getCols() const;
  Matrix operator+(const Matrix &other) const;
  Matrix operator*(const Matrix &other) const;
  Matrix operator*(double scalar) const;
  Matrix ReLU() const;
  Matrix transpose() const;
  Matrix operator-(const Matrix &other) const;
  Matrix elementwise(const Matrix &other) const;

private:
  int rows;
  int cols;
  std::vector<std::vector<double>> data;
};