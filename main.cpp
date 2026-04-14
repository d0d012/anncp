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

  } catch (const std::exception &e) {
    std::cerr << "\nKritik Hata: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}