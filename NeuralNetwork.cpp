#include "NeuralNetwork.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

Matrix NeuralNetwork::meanSquaredError(const Matrix &y_true,
                                       const Matrix &y_pred) {
  Matrix diff = y_true - y_pred;
  return diff.elementwise(diff) * 0.5;
}

NeuralNetwork::NeuralNetwork(
    const std::vector<int> &topology,
    const std::vector<Layer::Activation> &activations) {

  for (int i = 0; i < topology.size() - 1; i++) {
    Layer l = Layer(topology[i], topology[i + 1], activations[i]);
    layers.push_back(l);
  }
}
Layer &NeuralNetwork::getLayer(int index) {
  if (index < 0 || index >= layers.size()) {
    throw std::out_of_range("Layer index out of bounds");
  }
  return layers[index];
}
void NeuralNetwork::setLayer(int index, const Layer &l) {
  if (index < 0 || index >= layers.size()) {
    throw std::out_of_range("Layer index out of bounds");
  }
  layers[index] = l;
}

void NeuralNetwork::loadWeightsFromFile(const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open weights file: " + filepath);
  }

  double val;

  for (size_t i = 0; i < layers.size(); i++) {

    int wRows = layers[i].getWeights().getRows();
    int wCols = layers[i].getWeights().getCols();
    Matrix newW(wRows, wCols);

    for (int r = 0; r < wRows; r++) {
      for (int c = 0; c < wCols; c++) {
        if (file >> val) {
          newW.setValue(r, c, val);
        } else {
          throw std::runtime_error(
              "File ran out of numbers while reading weights for layer " +
              std::to_string(i));
        }
      }
    }
    layers[i].setWeights(newW);
    int bRows = layers[i].getBiases().getRows();
    int bCols = layers[i].getBiases().getCols();
    Matrix newB(bRows, bCols);

    for (int r = 0; r < bRows; r++) {
      for (int c = 0; c < bCols; c++) {
        if (file >> val) {
          newB.setValue(r, c, val);
        } else {
          throw std::runtime_error(
              "File ran out of numbers while reading biases for layer " +
              std::to_string(i));
        }
      }
    }
    layers[i].setBiases(newB);
  }

  file.close();
}

Matrix NeuralNetwork::predict(Matrix x) {
  for (size_t i = 0; i < layers.size(); i++) {
    x = layers[i].forward(x);
  }
  return x;
}
Matrix NeuralNetwork::train(Matrix x, Matrix y, double learning_rate,
                            int epochs) {
  for (int epoch = 0; epoch < epochs; epoch++) {
    Matrix prediction = predict(x);
    Matrix delta = prediction - y;
    for (int i = layers.size() - 1; i >= 0; i--) {
      delta = layers[i].backward(delta, learning_rate);
    }
    if (epoch % 2000 == 0) {
      Matrix current_loss = meanSquaredError(y, predict(x));
      double total = 0.0;
      for (int i = 0; i < current_loss.getRows(); i++)
        total += current_loss.getValue(i, 0);
      std::cout << "Epoch " << epoch << " loss: " << total / current_loss.getRows() << std::endl;
    }
  }
  return meanSquaredError(y, predict(x));
}
