#include "Matrix.h"
#include <stdexcept>

Matrix::Matrix(int rows, int cols)
    : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {};
double Matrix::getValue(int row, int col) const {
  if (row >= 0 && row < rows && col >= 0 && col < cols) {
    return data[row][col];
  }
  throw std::out_of_range("arguments are not valid");
};

void Matrix::setValue(int row, int col, double value) {
  if (row >= 0 && row < rows && col >= 0 && col < cols) {
    data[row][col] = value;
  } else {
    throw std::out_of_range("Matrix index out of bounds in setValue");
  }
}

int Matrix::getRows() const { return rows; };

int Matrix::getCols() const { return cols; };

Matrix Matrix::operator+(const Matrix &other) const {
  if (cols != other.cols || (rows != other.rows && other.rows != 1 && rows != 1)) {
    throw std::invalid_argument(
        "Matrices must have the same dimensions for addition.");
  }
  int outRows = std::max(rows, other.rows);
  Matrix result(outRows, cols);
  for (int i = 0; i < outRows; i++) {
    for (int j = 0; j < cols; j++) {
      result.data[i][j] = data[i % rows][j] + other.data[i % other.rows][j];
    }
  }
  return result;
}

Matrix Matrix::operator*(const Matrix &other) const {

  if (cols != other.rows) {
    throw std::invalid_argument(
        "Number of columns of the first matrix must be equal to the number of "
        "rows of the second matrix for multiplication.");
  }
  Matrix result(rows, other.cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < other.cols; j++) {
      double sum = 0.0;
      for (int k = 0; k < cols; k++) {
        sum += data[i][k] * other.data[k][j];
      }

      result.data[i][j] = sum;
    }
  }
  return result;
}
Matrix Matrix::operator*(double scalar) const {
  Matrix result(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result.data[i][j] = data[i][j] * scalar;
    }
  }
  return result;
}

Matrix Matrix::ReLU() const {
  Matrix res(rows, cols);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      double val = data[i][j];
      if (val > 0) {
        res.data[i][j] = val;
      } else {
        res.data[i][j] = 0.0;
      }
    }
  }
  return res;
}

Matrix Matrix::transpose() const {
  Matrix result(cols, rows);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result.data[j][i] = data[i][j];
    }
  }
  return result;
}

Matrix Matrix::operator-(const Matrix &other) const {
  if (rows != other.rows || cols != other.cols) {
    throw std::invalid_argument(
        "Matrices must have the same dimensions for subtraction.");
  }
  Matrix result(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result.data[i][j] = data[i][j] - other.data[i][j];
    }
  }
  return result;
}
Matrix Matrix::elementwise(const Matrix &other) const {
  Matrix result(rows, cols);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      result.setValue(i, j, data[i][j] * other.getValue(i, j));
  return result;
}