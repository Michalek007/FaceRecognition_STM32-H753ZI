//
// Created by Michał on 20.11.2024.
//
#include <stdlib.h>
#include <math.h>
#include "rnet.h"
#include "rnet_weights.h"
#include "cnn.h"


void RNet_Model(float* input, float* outputReg, float* outputProb){
    float output1[13552];
    CNN_ConvLayer(3, 24, 24, 28, 3, 3, 1, 1, 0, 0, input, rnet_weight0, rnet_bias0, output1);
    CNN_PReLU(28, 22, 22, output1, rnet_weight1);
    float output3[3388];
    CNN_MaxPool(28, 22, 22, 3, 3, 2, 2, 0, 0, 1, output1, output3);
    float output4[3888];
    CNN_ConvLayer(28, 11, 11, 48, 3, 3, 1, 1, 0, 0, output3, rnet_weight2, rnet_bias1, output4);
    CNN_PReLU(48, 9, 9, output4, rnet_weight3);
    float output6[768];
    CNN_MaxPool(48, 9, 9, 3, 3, 2, 2, 0, 0, 1, output4, output6);
    float output7[576];
    CNN_ConvLayer(48, 4, 4, 64, 2, 2, 1, 1, 0, 0, output6, rnet_weight4, rnet_bias2, output7);
    CNN_PReLU(64, 3, 3, output7, rnet_weight5);
    float permutedOutput8[576];
    CNN_Permute(64, 3, 3, 2, 1, 0, output7, permutedOutput8);
    float output9[128];
    CNN_FcLayer(576, 128, permutedOutput8, rnet_weight6, rnet_bias3, output9);
    CNN_PReLU(128, 1, 1, output9, rnet_weight7);

    CNN_FcLayer(128, 2, output9, rnet_weight8, rnet_bias4, outputProb);
    CNN_Softmax(2, outputProb);

    CNN_FcLayer(128, 4, output9, rnet_weight9, rnet_bias5, outputReg);
}
