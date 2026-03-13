#include "Layer.h"
#include <stdexcept>

Layer::Layer(int inputSize, int outputSize) : W(Matrix(inputSize,outputSize)),b(Matrix(1,outputSize)) {};

void Layer::setBiases(const Matrix& b){
    this->b=b;
}
Matrix Layer::getBiases(){
    return b;
}

void Layer::setWeights(const Matrix& W){
    this->W=W;
}
Matrix Layer::getWeights(){
    return W;
}

Matrix Layer::forward(const Matrix& x)const{
    return x*W+b;
}


