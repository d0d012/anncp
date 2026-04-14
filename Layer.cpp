#include "Layer.h"
#include <algorithm>
#include <cmath>
Layer::Layer(int inputSize, int outputSize, Activation act)
    : W(Matrix(inputSize, outputSize)), b(Matrix(1, outputSize)), act(act) {};

void Layer::setBiases(const Matrix &b) { this->b = b; }
Matrix Layer::getBiases() { return b; }

void Layer::setWeights(const Matrix &W) { this->W = W; }
Matrix Layer::getWeights() { return W; }

Matrix Layer::forward(const Matrix &x) const {
  Matrix z = x * W + b;
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
