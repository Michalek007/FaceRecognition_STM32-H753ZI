//
// Created by Michał on 15.10.2024.
//
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "cnn.h"

void CNN_FcLayer(size_t inputLen, size_t outputLen, const float* input, const float* weights, const float* biases, float* output){
   for (size_t i=0;i<outputLen;++i){
       float outputValue = 0;
       for (size_t j=0;j<inputLen;++j){
           outputValue += weights[i*inputLen+j]*input[j];
       }
       output[i] = outputValue + biases[i];
   }
}

void CNN_ConvLayer(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputChannels, size_t kernelHeight, size_t kernelWidth, int strideH, int strideW, int paddingH, int paddingW, const float* input, const float* weights, const float* biases, float* output){
    size_t outputHeight = (inputHeight-kernelHeight+2*paddingH)/strideH+1;
    size_t outputWidth = (inputWidth-kernelWidth+2*paddingW)/strideW+1;
    assert(outputHeight>0);
    assert(outputWidth>0);

    int hasPadding = 0;
    int paddingTop, paddingLeft, paddingRight, paddingBottom;
    size_t oldHeight, oldWidth;
    if (paddingH != 0 || paddingW != 0){
        oldHeight = inputHeight;
        oldWidth = inputWidth;
        int newHeight = inputHeight+2*paddingH;
        int newWidth = inputWidth+2*paddingW;
        hasPadding = 1;
        inputHeight = newHeight;
        inputWidth = newWidth;

        paddingTop = paddingH;
        paddingLeft = paddingW;
        paddingRight = paddingW;
        paddingBottom = paddingH;

    }

    for (size_t o=0;o<outputChannels;++o){
        for (size_t i=0;i<outputHeight;i++){
            for (size_t j=0;j<outputWidth;j++){
                float outputValue = 0;
                for (size_t p =0;p<inputChannels;p++){
                    for (size_t k=0;k<kernelHeight;k++){
                        for (size_t l=0;l<kernelWidth;l++){
                            int weightsIndex = o*kernelWidth*kernelHeight*inputChannels + p*kernelWidth*kernelHeight + k*kernelWidth +l;
                            int inputIndex = p*inputHeight*inputWidth + (i*strideH+k)*inputWidth + j*strideW + l;
                            float inputValue;
                            if (hasPadding){
                                inputIndex -= p*inputHeight*inputWidth;
                                size_t row = inputIndex / inputWidth;
                                size_t column = inputIndex % inputWidth;
                                if (row < paddingTop || row >= inputHeight - paddingBottom || column < paddingLeft || column >= inputWidth - paddingRight){
                                    inputValue = 0;
                                }
                                else{
                                    inputIndex = inputIndex - inputWidth*row - paddingLeft + oldWidth*(row-paddingTop); // last=> oldWidth*oldInputRow
                                    inputIndex += p*oldHeight*oldWidth;
                                    inputValue = input[inputIndex];
                                }
                            }
                            else{
                                inputValue = input[inputIndex];
                            }
                            outputValue += inputValue * weights[weightsIndex];
                        }
                    }
                }
                int outputIndex = o*outputHeight*outputWidth + i*outputWidth + j;
                output[outputIndex] = outputValue + biases[o];
            }
        }
    }
}

void CNN_ConvLayer_Symmetric(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputChannels, size_t kernelHeight, size_t kernelWidth,
                             int stride, int padding, const float* input, const float* weights, const float* biases, float* output){
    CNN_ConvLayer(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, stride, stride, padding, padding, input, weights, biases, output);
}

void CNN_ConvLayer_Basic(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputChannels, size_t kernel, const float* input, const float* weights, const float* biases, float* output){
    CNN_ConvLayer_Symmetric(inputChannels, inputHeight, inputWidth, outputChannels, kernel, kernel, 1, 0, input, weights, biases, output);
}

void CNN_ReLU(size_t inputLen, float* input){
    for (size_t i=0;i<inputLen;++i){
        if (input[i] < 0){
            input[i] = 0;
        }
    }
}

void CNN_MaxPool(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t kernelHeight, size_t kernelWidth, int strideH, int strideW, int paddingH, int paddingW, int ceilMode, const float* input, float* output){
    size_t outputHeight, outputWidth;
    if (ceilMode){
        outputHeight = ceilf((inputHeight-kernelHeight+2*paddingH)/(float)strideH+1);
        outputWidth = ceilf((inputWidth-kernelWidth+2*paddingW)/(float)strideW+1);
    }
    else{
        outputHeight = (inputHeight-kernelHeight+2*paddingH)/strideH+1;
        outputWidth = (inputWidth-kernelWidth+2*paddingW)/strideW+1;
    }
    assert(outputHeight>0);
    assert(outputWidth>0);

    int paddingRight = 0;
    int paddingBottom = 0;
    if (ceilMode){
        paddingRight = (outputWidth-1)*strideW + kernelWidth - inputWidth;
        paddingBottom = (outputHeight-1)*strideH + kernelHeight - inputHeight;
    }

    int hasPadding = 0;
    int paddingTop, paddingLeft;
    size_t oldHeight, oldWidth;
    if (paddingH != 0 || paddingW != 0 || paddingRight !=0 || paddingBottom != 0){
        oldHeight = inputHeight;
        oldWidth = inputWidth;
        int newHeight = inputHeight+2*paddingH + paddingBottom;
        int newWidth = inputWidth+2*paddingW + paddingRight;
        hasPadding = 1;
        inputHeight = newHeight;
        inputWidth = newWidth;

        paddingTop = paddingH;
        paddingLeft = paddingW;
        paddingRight += paddingW;
        paddingBottom += paddingH;
    }

    for (size_t o=0;o<inputChannels;++o){
        for (size_t i=0;i<outputHeight;i++){
            for (size_t j=0;j<outputWidth;j++){
                float maxValue = 0;
                for (size_t k=0;k<kernelHeight;k++){
                    for (size_t l=0;l<kernelWidth;l++){
                        int inputIndex = o*inputHeight*inputWidth + (i*strideH+k)*inputWidth + j*strideW + l;
                        float targetValue;
                        if (hasPadding){
                            inputIndex -= o*inputHeight*inputWidth;
                            size_t row = inputIndex / inputWidth;
                            size_t column = inputIndex % inputWidth;
                            if (row < paddingTop || row >= inputHeight - paddingBottom || column < paddingLeft || column >= inputWidth - paddingRight){
                                targetValue = -FLT_MAX;
                            }
                            else{
                                inputIndex = inputIndex - inputWidth*row - paddingLeft + oldWidth*(row-paddingTop); // last=> oldWidth*oldInputRow
                                inputIndex += o*oldHeight*oldWidth;
                                targetValue = input[inputIndex];
                            }
                        }
                        else{
                            targetValue = input[inputIndex];
                        }
                        if (maxValue < targetValue || (k == 0 && l == 0)){
                            maxValue = targetValue;
                        }
                    }
                }
                int outputIndex = o*outputHeight*outputWidth+i*outputWidth + j;
                output[outputIndex] = maxValue;
            }
        }
    }
}

void CNN_MaxPool_Symmetric(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t kernelHeight, size_t kernelWidth, int stride, int padding, const float* input, float* output){
    CNN_MaxPool(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, stride, stride, padding, padding, 0, input, output);
}

void CNN_MaxPool_Basic(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t kernel, const float* input, float* output){
    CNN_MaxPool_Symmetric(inputChannels, inputHeight, inputWidth, kernel, kernel, (int) kernel, 0, input, output);
}

void CNN_PReLU(size_t inputChannels, size_t inputHeight, size_t inputWidth, float* input, const float* weights){
    for (size_t o=0;o<inputChannels;++o){
        for (size_t i=0;i<inputHeight;++i){
            for (size_t j=0;j<inputWidth;++j){
                size_t index = o*inputHeight*inputWidth + i*inputWidth + j;
                if (input[index] < 0){
                    input[index] = weights[o] * input[index];
                }
            }
        }
    }
}

void CNN_Softmax(size_t inputLen, float* input){
    float sum = 0;
    for (size_t i=0;i<inputLen;++i){
        input[i] = expf(input[i]);
        sum += input[i];
    }
    for (size_t i=0;i<inputLen;++i){
        input[i] = input[i] / sum;
    }
}

void CNN_Softmax2D(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t dim, float* input){
    size_t firstLoop, secondLoop, thirdLoop;
    switch (dim){
        case 0:
            firstLoop = inputHeight;
            secondLoop = inputWidth;
            thirdLoop = inputChannels;
            break;
        case 1:
            firstLoop = inputHeight;
            secondLoop = inputWidth;
            thirdLoop = inputChannels;
            break;
        default:
            firstLoop = inputChannels;
            secondLoop = inputHeight;
            thirdLoop = inputWidth;
    }

    for (size_t o=0;o<firstLoop;++o){
        for (size_t i=0;i<secondLoop;++i){
            float sum = 0;
            size_t step = 0;
            while (step<2){
                for (size_t j=0;j<thirdLoop;++j){
                    size_t index;
                    switch (dim){
                        case 0:
                            index =  j*inputHeight*inputWidth + o*inputWidth + i;
                            break;
                        case 1:
                            index =  o*inputHeight*inputWidth + j*inputWidth + i;
                            break;
                        default:
                            index =  o*inputHeight*inputWidth + i*inputWidth + j;
                    }

                    if (step == 0){
                        input[index] = expf(input[index]);
                        sum += input[index];
                    }
                    else if (step == 1){
                        input[index] = input[index] / sum;
                    }
                }
                ++step;
            }
        }
    }
}

void CNN_Permute(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputChannelsDim, size_t outputHeightDim, size_t outputWidthDim, const float* input, float* output){
//    for (size_t o=0;o<inputChannels;++o){
//        for (size_t i=0;i<inputHeight;++i){
//            for (size_t j=0;j<inputWidth;++j){
//                size_t inputIndex = o*inputHeight*inputWidth + i*inputWidth + j;
//                size_t outputIndex = o*inputHeight*inputWidth + i*inputWidth + j;
//            }
//        }
//    }
    size_t outputChannels = inputChannels;
    size_t outputHeight = inputHeight;
    size_t outputWidth = inputWidth;
    if (outputChannelsDim == 2){
        outputChannels = inputWidth;
    }
    else if (outputChannelsDim == 1){
        outputChannels = inputHeight;
    }

    if (outputHeightDim == 2){
        outputHeight = inputWidth;
    }
    else if (outputHeightDim == 0){
        outputHeight = inputChannels;
    }

    if (outputWidthDim == 1){
        outputWidth = inputHeight;
    }
    else if (outputWidthDim == 0){
        outputWidth = inputChannels;
    }


    for (size_t o=0;o<outputChannels;++o){
        for (size_t i=0;i<outputHeight;++i){
            for (size_t j=0;j<outputWidth;++j){
                size_t factor_o, factor_i, factor_j;
                factor_o = o;
                factor_i = i;
                factor_j = j;

                if (outputChannelsDim == 2){
                    factor_o = j;
                }
                else if (outputChannelsDim == 1){
                    factor_o = i;
                }

                if (outputHeightDim == 2){
                    factor_i = j;
                }
                else if (outputHeightDim == 0){
                    factor_i = o;
                }

                if (outputWidthDim == 1){
                    factor_j = i;
                }
                else if (outputWidthDim == 0){
                    factor_j = o;
                }

                size_t outputIndex = o*outputHeight*outputWidth + i*outputWidth + j;
                size_t inputIndex = factor_o*inputHeight*inputWidth + factor_i*inputWidth + factor_j;
                output[outputIndex] = input[inputIndex];
            }
        }
    }
}

/*
 * Boxes are expected to be in format: x, y, x2, y2 where: x2 > x, y2 > y
 */
void CNN_BoxIou(size_t boxesLen, const float* boxes, size_t boxesLen2, const float* boxes2, float* output){
    for (size_t i=0;i<boxesLen*4;i+=4){
        float area = (boxes[i+2] - boxes[i]) * (boxes[i+3] - boxes[i+1]);
        for (size_t j=0;j<boxesLen2*4;j+=4){
            float x = fmaxf(0, fminf(boxes[i+2], boxes2[j+2]) - fmaxf(boxes[i], boxes2[j]));
            float y = fmaxf(0, fminf(boxes[i+3], boxes2[j+3]) - fmaxf(boxes[i+1], boxes2[j+1]));
            float overlapArea = x * y;
            float area2 = (boxes2[j+2] - boxes2[j]) * (boxes2[j+3] - boxes2[j+1]);
            size_t index = i/4*boxesLen2+j/4;
            output[index] = overlapArea / (area + area2 - overlapArea);
//            output[index] = CNN_Iou(boxes[i], boxes[i+1], boxes[i+2], boxes[i+3], boxes2[j], boxes2[j+1], boxes2[j+2], boxes2[j+3]);
        }
    }
}

float CNN_Iou(float x, float y, float x2, float y2, float xp, float yp, float x2p, float y2p){
    float area = (x2 - x) * (y2 - y);
    float area2 = (x2p - xp) * (y2p - yp);
    float x_distance = fmaxf(0, fminf(x2, x2p) - fmaxf(x, xp));
    float y_distance = fmaxf(0, fminf(y2, y2p) - fmaxf(y, yp));
    float overlapArea = x_distance * y_distance;
    return overlapArea / (area + area2 - overlapArea);
}

int CNN_BoxNms(size_t boxesLen, const float* boxes, const float* scores, float iouThreshold, float* output){
    int indexes[boxesLen];
    for (size_t i=0;i<boxesLen;++i){
        indexes[i] = -1;
    }
    for (size_t i=0;i<boxesLen*4;i+=4){
        if (indexes[i/4] == -2)
            continue;
        for (size_t j=i+4;j<boxesLen*4;j+=4){
            if (indexes[j/4] == -2)
                continue;

            float iou = CNN_Iou(boxes[i], boxes[i+1], boxes[i+2], boxes[i+3], boxes[j], boxes[j+1], boxes[j+2], boxes[j+3]);
            int boxI = i;
            int boxJ = j;

            if (iou > iouThreshold){
                if (scores[i/4] >= scores[j/4]){
                    boxJ = -2;
                }
                else{
                    boxI = -2;
                }
            }
            indexes[i/4] = boxI;
            indexes[j/4] = boxJ;
            if (boxI == -2)
                break;
        }
    }
    size_t outputIndex = 0;
    for (size_t i=0;i<boxesLen;++i){
        int inputIndex = indexes[i];
        if (inputIndex < 0)
            continue;
        output[outputIndex] = boxes[inputIndex];
        output[outputIndex+1] = boxes[inputIndex+1];
        output[outputIndex+2] = boxes[inputIndex+2];
        output[outputIndex+3] = boxes[inputIndex+3];
        outputIndex += 4;
    }
    return outputIndex/4;
}

int CNN_BoxNmsIdx(size_t boxesLen, const float* boxes, const float* scores, float iouThreshold, int* boxesIndexes){
    int indexes[boxesLen];
    for (size_t i=0;i<boxesLen;++i){
        indexes[i] = -1;
    }
    for (size_t i=0;i<boxesLen*4;i+=4){
        if (indexes[i/4] == -2)
            continue;
        for (size_t j=i+4;j<boxesLen*4;j+=4){
            if (indexes[j/4] == -2)
                continue;

            float iou = CNN_Iou(boxes[i], boxes[i+1], boxes[i+2], boxes[i+3], boxes[j], boxes[j+1], boxes[j+2], boxes[j+3]);
            int boxI = i;
            int boxJ = j;

            if (iou > iouThreshold){
                if (scores[i/4] >= scores[j/4]){
                    boxJ = -2;
                }
                else{
                    boxI = -2;
                }
            }
            indexes[i/4] = boxI;
            indexes[j/4] = boxJ;
            if (boxI == -2)
                break;
        }
    }
    size_t outputIndex = 0;
    for (size_t i=0;i<boxesLen;++i){
        int inputIndex = indexes[i];
        if (inputIndex < 0)
            continue;
        boxesIndexes[outputIndex] = inputIndex/4;
        ++outputIndex;
    }
    return outputIndex;
}

void CNN_AdaptiveAveragePool(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputHeight, size_t outputWidth, const float* input, float* output){

    for (size_t o=0;o<inputChannels;++o){
        for (size_t i=0;i<outputHeight;i++){
            for (size_t j=0;j<outputWidth;j++){
                float sum = 0;
                int startIndexH = i * inputHeight/outputHeight;
                int startIndexW = j * inputWidth/outputWidth;
                int endIndexH = ceilf((i+1) * inputHeight/(float)outputHeight);
                int endIndexW = ceilf((j+1) * inputWidth/(float)outputWidth);
                size_t kernelHeight = endIndexH - startIndexH;
                size_t kernelWidth = endIndexW - startIndexW;
                for (size_t k=0;k<kernelHeight;k++){
                    for (size_t l=0;l<kernelWidth;l++){
                        int inputIndex = o*inputHeight*inputWidth + (startIndexH+k)*inputWidth + startIndexW + l;
                        sum += input[inputIndex];
                    }
                }
                int outputIndex = o*outputHeight*outputWidth+i*outputWidth + j;
                output[outputIndex] = sum / (kernelHeight * kernelWidth);
            }
        }
    }
}

void CNN_AdaptiveAveragePool_Uint8_Float(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputHeight, size_t outputWidth, const uint8_t * input, float* output){

    for (size_t o=0;o<inputChannels;++o){
        for (size_t i=0;i<outputHeight;i++){
            for (size_t j=0;j<outputWidth;j++){
                float sum = 0;
                int startIndexH = i * inputHeight/outputHeight;
                int startIndexW = j * inputWidth/outputWidth;
                int endIndexH = ceilf((i+1) * inputHeight/(float)outputHeight);
                int endIndexW = ceilf((j+1) * inputWidth/(float)outputWidth);
                size_t kernelHeight = endIndexH - startIndexH;
                size_t kernelWidth = endIndexW - startIndexW;
                for (size_t k=0;k<kernelHeight;k++){
                    for (size_t l=0;l<kernelWidth;l++){
                        int inputIndex = o*inputHeight*inputWidth + (startIndexH+k)*inputWidth + startIndexW + l;
                        sum += input[inputIndex];
                    }
                }
                int outputIndex = o*outputHeight*outputWidth+i*outputWidth + j;
                output[outputIndex] = sum / (kernelHeight * kernelWidth);
            }
        }
    }
}

void CNN_AdaptiveAveragePool_Uint8_Uint8(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputHeight, size_t outputWidth, const uint8_t * input, uint8_t* output){
    for (size_t o=0;o<inputChannels;++o){
        for (size_t i=0;i<outputHeight;i++){
            for (size_t j=0;j<outputWidth;j++){
                float sum = 0;
                int startIndexH = i * inputHeight/outputHeight;
                int startIndexW = j * inputWidth/outputWidth;
                int endIndexH = ceilf((i+1) * inputHeight/(float)outputHeight);
                int endIndexW = ceilf((j+1) * inputWidth/(float)outputWidth);
                size_t kernelHeight = endIndexH - startIndexH;
                size_t kernelWidth = endIndexW - startIndexW;
                for (size_t k=0;k<kernelHeight;k++){
                    for (size_t l=0;l<kernelWidth;l++){
                        int inputIndex = o*inputHeight*inputWidth + (startIndexH+k)*inputWidth + startIndexW + l;
                        sum += input[inputIndex];
                    }
                }
                int outputIndex = o*outputHeight*outputWidth+i*outputWidth + j;
                output[outputIndex] = (int)roundf(sum / (kernelHeight * kernelWidth));
            }
        }
    }
}

void CNN_AveragePool(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t kernelHeight, size_t kernelWidth, int strideH, int strideW, int paddingH, int paddingW, int ceilMode, const float* input, float* output){
    size_t outputHeight, outputWidth;
    if (ceilMode){
        outputHeight = ceilf((inputHeight-kernelHeight+2*paddingH)/(float)strideH+1);
        outputWidth = ceilf((inputWidth-kernelWidth+2*paddingW)/(float)strideW+1);
    }
    else{
        outputHeight = (inputHeight-kernelHeight+2*paddingH)/strideH+1;
        outputWidth = (inputWidth-kernelWidth+2*paddingW)/strideW+1;
    }
    assert(outputHeight>0);
    assert(outputWidth>0);

    int paddingRight = 0;
    int paddingBottom = 0;
    if (ceilMode){
        paddingRight = (outputWidth-1)*strideW + kernelWidth - inputWidth;
        paddingBottom = (outputHeight-1)*strideH + kernelHeight - inputHeight;
    }

    int hasPadding = 0;
    int paddingTop, paddingLeft;
    size_t oldHeight, oldWidth;
    if (paddingH != 0 || paddingW != 0 || paddingRight !=0 || paddingBottom != 0){
        oldHeight = inputHeight;
        oldWidth = inputWidth;
        int newHeight = inputHeight+2*paddingH + paddingBottom;
        int newWidth = inputWidth+2*paddingW + paddingRight;
        hasPadding = 1;
        inputHeight = newHeight;
        inputWidth = newWidth;

        paddingTop = paddingH;
        paddingLeft = paddingW;
        paddingRight += paddingW;
        paddingBottom += paddingH;
    }

    for (size_t o=0;o<inputChannels;++o){
        for (size_t i=0;i<outputHeight;i++){
            for (size_t j=0;j<outputWidth;j++){
                float sum = 0;
                for (size_t k=0;k<kernelHeight;k++){
                    for (size_t l=0;l<kernelWidth;l++){
                        int inputIndex = o*inputHeight*inputWidth + (i*strideH+k)*inputWidth + j*strideW + l;
                        float targetValue;
                        if (hasPadding){
                            inputIndex -= o*inputHeight*inputWidth;
                            size_t row = inputIndex / inputWidth;
                            size_t column = inputIndex % inputWidth;
                            if (row < paddingTop || row >= inputHeight - paddingBottom || column < paddingLeft || column >= inputWidth - paddingRight){
                                targetValue = 0;
                            }
                            else{
                                inputIndex = inputIndex - inputWidth*row - paddingLeft + oldWidth*(row-paddingTop); // last=> oldWidth*oldInputRow
                                inputIndex += o*oldHeight*oldWidth;
                                targetValue = input[inputIndex];
                            }
                        }
                        else{
                            targetValue = input[inputIndex];
                        }
                        sum += targetValue;
                    }
                }
                int outputIndex = o*outputHeight*outputWidth+i*outputWidth + j;
                output[outputIndex] = sum / (kernelHeight * kernelWidth);
            }
        }
    }
}

void CNN_AveragePool_Symmetric(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t kernelHeight, size_t kernelWidth, int stride, int padding, const float* input, float* output){
    CNN_AveragePool(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, stride, stride, padding, padding, 0, input, output);
}

void CNN_AveragePool_Basic(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t kernel, const float* input, float* output){
    CNN_AveragePool_Symmetric(inputChannels, inputHeight, inputWidth, kernel, kernel, (int) kernel, 0, input, output);
}

void CNN_LeakyReLU(size_t inputChannels, size_t inputHeight, size_t inputWidth, float* input, const float weight){
    for (size_t o=0;o<inputChannels;++o){
        for (size_t i=0;i<inputHeight;++i){
            for (size_t j=0;j<inputWidth;++j){
                size_t index = o*inputHeight*inputWidth + i*inputWidth + j;
                if (input[index] < 0){
                    input[index] = weight * input[index];
                }
            }
        }
    }
}

void CNN_BatchNorm(size_t inputChannels, size_t inputHeight, size_t inputWidth, float* input, const float* weights, const float* biases, const float* means, const float* variances){
    for (size_t o=0;o<inputChannels;++o){
        for (size_t i=0;i<inputHeight;++i){
            for (size_t j=0;j<inputWidth;++j){
                size_t index = o*inputHeight*inputWidth + i*inputWidth + j;
                input[index] = (input[index] - means[o]) / sqrtf(variances[o]+1e-05f) * weights[o] + biases[o];
            }
        }
    }
}

void CNN_Normalize(size_t inputChannels, size_t inputHeight, size_t inputWidth, float* input, const float* means, const float* stds){
    for (size_t o=0;o<inputChannels;++o){
        for (size_t i=0;i<inputHeight;++i){
            for (size_t j=0;j<inputWidth;++j){
                size_t index = o*inputHeight*inputWidth + i*inputWidth + j;
                input[index] = (input[index] - means[o]) / stds[o];
            }
        }
    }
}

void CNN_NormalizeLp(size_t inputLen, float p, float* input){
    float lpNorm = 0;
    for (size_t i=0;i<inputLen;++i){
        lpNorm += powf(input[i], p);
    }
    lpNorm = fmaxf(powf(lpNorm, 1/p), 1e-12);
    for (size_t i=0;i<inputLen;++i){
        input[i] /= lpNorm;
    }
}
