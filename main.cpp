#include "Matrix.h"
#include "NeuralNetwork.h"
#include <iostream>
#include <vector>

int main() {
  try {

    std::vector<int> topology = {2, 4, 1};
    std::vector<Layer::Activation> activations = {
        Layer::Activation::ReLU,
        Layer::Activation::Sigmoid,
    };
    NeuralNetwork net(topology, activations);

    std::cout << "Agirliklar weights.txt dosyasindan yukleniyor..."
              << std::endl;
    net.loadWeightsFromFile("weights.txt");
    std::cout << "Agirliklar basariyla yuklendi!\n" << std::endl;

    Matrix x(1, 2);
    x.setValue(0, 0, 1.0);
    x.setValue(0, 1, 0.5);

    std::cout << "Girdi (x): [" << x.getValue(0, 0) << ", " << x.getValue(0, 1)
              << "]" << std::endl;

    Matrix prediction = net.predict(x);

    std::cout << "Tahmin (y): " << prediction.getValue(0, 0) << std::endl;

    // XOR training
    Matrix X(4, 2);
    X.setValue(0, 0, 0);
    X.setValue(0, 1, 0);
    X.setValue(1, 0, 0);
    X.setValue(1, 1, 1);
    X.setValue(2, 0, 1);
    X.setValue(2, 1, 0);
    X.setValue(3, 0, 1);
    X.setValue(3, 1, 1);

    Matrix Y(4, 1);
    Y.setValue(0, 0, 0);
    Y.setValue(1, 0, 1);
    Y.setValue(2, 0, 1);
    Y.setValue(3, 0, 0);

    NeuralNetwork xor_net(
        {2, 4, 1}, {Layer::Activation::Sigmoid, Layer::Activation::Sigmoid});
    xor_net.loadWeightsFromFile("weights.txt");
    Matrix xor_prediction = xor_net.predict(X);
    for (int i = 0; i < xor_prediction.getRows(); i++) {
      std::cout << "Prediction: " << xor_prediction.getValue(i, 0) << std::endl;
    }

    Matrix loss = xor_net.train(X, Y, 0.1, 10000);
    std::cout << "\nXOR final loss: " << loss.getValue(0, 0) << std::endl;
    xor_prediction = xor_net.predict(X);
    for (int i = 0; i < xor_prediction.getRows(); i++) {
      std::cout << "Prediction: " << xor_prediction.getValue(i, 0) << std::endl;
    }

  } catch (const std::exception &e) {
    std::cerr << "\nKritik Hata: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}