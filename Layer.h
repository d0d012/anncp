#pragma once
#include "Matrix.h"
class Layer {
public:
  enum class Activation { ReLU, Sigmoid, Tanh, LeakyReLU, None };
  Layer(int inputSize, int outputSize, Activation act);

  Matrix forward(const Matrix &x);
  Matrix backward(const Matrix &delta_in, double learning_rate);
  void setBiases(const Matrix &b);

  Matrix getBiases();
  void setWeights(const Matrix &W);
  Matrix getWeights();

private:
  Matrix W;
  Matrix b;
  Matrix last_input;
  Matrix last_z;
  Activation act;
  double activation_function(double x) const;
  double activation_derivative_scalar(double x) const;
  Matrix activation_derivative(const Matrix &z) const;
};
