#pragma once 
#include <vector>
#include "Layer.h"
#include "Matrix.h" 

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& topology);
    
    Layer& getLayer(int index); 
    void setLayer(int index, const Layer& l);
    
    Matrix predict( Matrix x) const;

    void loadWeightsFromFile(const std::string& filepath);

private:
     std::vector<Layer> layers;
};