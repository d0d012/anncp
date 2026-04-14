#pragma once
#include "Layer.h"
#include "Matrix.h"
#include <vector>

class NeuralNetwork {
public:
  NeuralNetwork(const std::vector<int> &topology,
                const std::vector<Layer::Activation> &activations);

  Layer &getLayer(int index);
  void setLayer(int index, const Layer &l);

  Matrix predict(Matrix x);
  Matrix train(Matrix x, Matrix y, double learning_rate, int epochs);
  Matrix meanSquaredError(const Matrix &y_true, const Matrix &y_pred);

  void loadWeightsFromFile(const std::string &filepath);

private:
  std::vector<Layer> layers;
};