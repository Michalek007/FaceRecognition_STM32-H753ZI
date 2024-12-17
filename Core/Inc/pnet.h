//
// Created by Michał on 19.11.2024.
//

#ifndef CNN_PNET_H
#define CNN_PNET_H

#include <stdio.h>

void PNet_Model(size_t inputChannels, size_t inputHeight, size_t inputWidth, float* input, float* outputReg, float* outputProb);

size_t PNet_GetOutputRegHeight(size_t inputHeight);
size_t PNet_GetOutputRegWidth(size_t inputWidth);
size_t PNet_GetOutputRegSize(size_t inputHeight, size_t inputWidth);

size_t PNet_GetOutputProbSize(size_t inputHeight, size_t inputWidth);

#endif //CNN_PNET_H
