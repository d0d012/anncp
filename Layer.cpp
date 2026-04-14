#include "Layer.h"
#include <algorithm>
#include <cmath>
Layer::Layer(int inputSize, int outputSize, Activation act)
    : W(Matrix(inputSize, outputSize)), b(Matrix(1, outputSize)), act(act),
      last_input(), last_z() {};

void Layer::setBiases(const Matrix &b) { this->b = b; }
Matrix Layer::getBiases() { return b; }

void Layer::setWeights(const Matrix &W) { this->W = W; }
Matrix Layer::getWeights() { return W; }

Matrix Layer::forward(const Matrix &x) {
  Matrix z = x * W + b;
  last_input = x;
  last_z = z;

  for (int r = 0; r < z.getRows(); ++r) {
    for (int c = 0; c < z.getCols(); ++c) {
      z.setValue(r, c, activation_function(z.getValue(r, c)));
    }
  }
  return z;
}

double Layer::activation_function(double x) const {
  switch (act) {
  case Activation::ReLU:
    return std::max(0.0, x);
  case Activation::Sigmoid:
    return 1.0 / (1.0 + std::exp(-x));
  case Activation::Tanh:
    return std::tanh(x);
  case Activation::LeakyReLU:
    return std::max(0.01 * x, x);
  case Activation::None:
    return x;
  default:
    return x;
  }
}
double Layer::activation_derivative_scalar(double x) const {
  switch (act) {
  case Activation::Sigmoid: {
    double s = activation_function(x);
    return s * (1 - s);
  }
  case Activation::ReLU:
    return x > 0 ? 1.0 : 0.0;
  case Activation::Tanh: {
    double t = std::tanh(x);
    return 1 - t * t;
  }
  case Activation::LeakyReLU:
    return x > 0 ? 1.0 : 0.01;
  case Activation::None:
    return 1.0;
  }
}
Matrix Layer::activation_derivative(const Matrix &z) const {
  Matrix result(z.getRows(), z.getCols());
  for (int r = 0; r < z.getRows(); r++)
    for (int c = 0; c < z.getCols(); c++)
      result.setValue(r, c, activation_derivative_scalar(z.getValue(r, c)));
  return result;
}

Matrix Layer::backward(const Matrix &delta_in, double learning_rate) {
  Matrix delta = delta_in.elementwise(activation_derivative(last_z));
  Matrix delta_out = delta * W.transpose();
  Matrix dE_dW = last_input.transpose() * delta;
  Matrix dE_db(1, delta.getCols());
  for (int r = 0; r < delta.getRows(); r++)
    for (int c = 0; c < delta.getCols(); c++)
      dE_db.setValue(0, c, dE_db.getValue(0, c) + delta.getValue(r, c));
  W = W - dE_dW * learning_rate;
  b = b - dE_db * learning_rate;
  return delta_out;
}