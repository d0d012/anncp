#pragma once
#include "Matrix.h"
class Layer {
public:
  enum class Activation { ReLU, Sigmoid, Tanh, LeakyReLU, None };
  Layer(int inputSize, int outputSize, Activation act);

  Matrix forward(const Matrix &x) const;
  void setBiases(const Matrix &b);

  Matrix getBiases();
  void setWeights(const Matrix &W);
  Matrix getWeights();

private:
  Matrix W;
  Matrix b;
  Activation act;
  double activation_function(double x) const;
};
