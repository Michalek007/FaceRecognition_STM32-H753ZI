//
// Created by Michał on 19.11.2024.
//

#include <stdlib.h>
#include <math.h>
#include "pnet.h"
#include "pnet_weights.h"
#include "cnn.h"

void PNet_Model(size_t inputChannels, size_t inputHeight, size_t inputWidth, float* input, float* outputReg, float* outputProb){

    size_t newHeight = inputHeight - 2;
    size_t newWidth = inputWidth - 2;
    float output1[10*newHeight*newWidth];
    CNN_ConvLayer(inputChannels, inputHeight, inputWidth, 10, 3, 3, 1, 1, 0, 0, input, pnet_weight0, pnet_bias0,
                  output1);
    inputHeight = newHeight;
    inputWidth = newWidth;

    CNN_PReLU(10, inputHeight, inputWidth, output1, pnet_weight1);

    newHeight = ceilf(inputHeight/2.0f);
    newWidth = ceilf(inputWidth/2.0f);
    float output3[10*newHeight*newWidth];
    CNN_MaxPool(10, inputHeight, inputWidth, 2, 2, 2, 2, 0, 0, 1, output1, output3);
    inputHeight = newHeight;
    inputWidth = newWidth;

    newHeight = inputHeight - 2;
    newWidth = inputWidth - 2;
    float output4[16*newHeight*newWidth];
    CNN_ConvLayer(10, inputHeight, inputWidth, 16, 3, 3, 1, 1, 0, 0, output3, pnet_weight2, pnet_bias1, output4);
    inputHeight = newHeight;
    inputWidth = newWidth;

    CNN_PReLU(16, inputHeight, inputWidth, output4, pnet_weight3);

    newHeight = inputHeight - 2;
    newWidth = inputWidth - 2;
    float output6[32*newHeight*newWidth];
    CNN_ConvLayer(16, inputHeight, inputWidth, 32, 3, 3, 1, 1, 0, 0, output4, pnet_weight4, pnet_bias2, output6);
    inputHeight = newHeight;
    inputWidth = newWidth;

    CNN_PReLU(32, inputHeight, inputWidth, output6, pnet_weight5);

    CNN_ConvLayer(32, inputHeight, inputWidth, 2, 1, 1, 1, 1, 0, 0, output6, pnet_weight6, pnet_bias3, outputProb);
    CNN_Softmax2D(2, inputHeight, inputWidth, 0, outputProb);

    CNN_ConvLayer(32, inputHeight, inputWidth, 4, 1, 1, 1, 1, 0, 0, output6, pnet_weight7, pnet_bias4, outputReg);
}

size_t PNet_GetOutputRegSize(size_t inputHeight, size_t inputWidth){
    size_t outputHeight = ceilf((inputHeight - 2) / 2.0f) - 4;
    size_t outputWidth = ceilf((inputWidth - 2) / 2.0f) - 4;
    size_t outputChannels = 4;
    return outputChannels * outputHeight * outputWidth;
}

size_t PNet_GetOutputProbSize(size_t inputHeight, size_t inputWidth){
    size_t outputHeight = ceilf((inputHeight - 2) / 2.0f) - 4;
    size_t outputWidth = ceilf((inputWidth - 2) / 2.0f) - 4;
    size_t outputChannels = 2;
    return outputChannels * outputHeight * outputWidth;
}

size_t PNet_GetOutputRegHeight(size_t inputHeight){
    size_t outputHeight = ceilf((inputHeight - 2) / 2.0f) - 4;
    return outputHeight;
}

size_t PNet_GetOutputRegWidth(size_t inputWidth){
    size_t outputWidth = ceilf((inputWidth - 2) / 2.0f) - 4;
    return outputWidth;
}
