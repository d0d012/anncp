#pragma once
#include "Matrix.h"
class Layer{
    public:
    Layer(int inputSize, int outputSize);

    Matrix forward(const Matrix& x)const;
    void setBiases(const Matrix& b);

    Matrix getBiases();
    void setWeights(const Matrix& W);
    Matrix getWeights();

    private:
    Matrix W;
    Matrix b;
    
};