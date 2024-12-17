//
// Created by Michał on 15.11.2024.
//

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "mtcnn.h"
#include "cnn.h"
#include "pnet.h"
#include "rnet.h"


int MTCNN_DetectFace(size_t inputChannels, size_t inputHeight, size_t inputWidth, const uint8_t* input, float* output){
    const float thresholdPNet = 0.6;
    const float thresholdRNet = 0.7;
    const float minSize = 20;
    const float factor = 0.709;
    const float iouThresholdPNet0 = 0.5;
    const float iouThresholdPNet1 = 0.5;
    const float iouThresholdRNet = 0.1;

    // scale pyramid
    float m = 12.0 / minSize;
    float minL = fminf(inputHeight, inputWidth) * m;
    float scaleI = m;
    size_t maxScaleSize = ceilf(logf(12/minL)/logf(factor));
    float scales[maxScaleSize];
    size_t scalesSize;
    for (scalesSize=0;minL>=12.0;minL*=factor){
        scales[scalesSize] = scaleI;
        scaleI *= factor;
        ++scalesSize;
    }

    size_t maxBoxesPerScale = 10; // should be given as argument
    size_t boxesMaxSize = scalesSize*9*maxBoxesPerScale; // or this
    float boxes[boxesMaxSize];
    size_t currentBoxesCount = 0;
    for (size_t i=0;i<scalesSize;++i){
        size_t outputHeight = inputHeight*scales[i]+1;
        size_t outputWidth = inputWidth*scales[i]+1;
        float scaledOutput[inputChannels*outputHeight*outputWidth];
        CNN_AdaptiveAveragePool_Uint8(inputChannels, inputHeight, inputWidth, outputHeight, outputWidth, input, scaledOutput);
//        if (i == 0){
//            for (size_t j=0;j<11163;++j){
//                printf("Output [%d]: %f\n", j, scaledOutput[j]);
//                assert(equalFloatDefault(scaledOutput[j], expectedOutput0[j]));
//            }
//        }
//        else if (i == 1){
//            for (size_t j=0;j<5547;++j){
//                printf("Output [%d]: %f\n", j, scaledOutput[j]);
//                assert(equalFloatDefault(scaledOutput[j], expectedOutput1[j]));
//            }
//        }
//        else if (i == 2){
//            for (size_t j=0;j<2883;++j){
//                printf("Output [%d]: %f\n", j, scaledOutput[j]);
//                assert(equalFloatDefault(scaledOutput[j], expectedOutput2[j]));
//            }
//        }
//        else if (i == 3){
//            for (size_t j=0;j<1452;++j){
//                printf("Output [%d]: %f\n", j, scaledOutput[j]);
//                assert(equalFloatDefault(scaledOutput[j], expectedOutput3[j]));
//            }
//        }
//        else if (i == 4){
//            for (size_t j=0;j<768;++j){
//                printf("Output [%d]: %f\n", j, scaledOutput[j]);
//                assert(equalFloatDefault(scaledOutput[j], expectedOutput4[j]));
//            }
//        }
        for (size_t j=0;j<inputChannels*outputHeight*outputWidth;++j){
            scaledOutput[j] = (scaledOutput[j] - 127.5f) * 0.0078125f;
        }
        size_t outputRegSize = PNet_GetOutputRegSize(outputHeight, outputWidth);
        size_t outputProbSize = PNet_GetOutputProbSize(outputHeight, outputWidth);
        float outputReg[outputRegSize];
        float outputProb[outputProbSize];
        PNet_Model(inputChannels, outputHeight, outputWidth, scaledOutput, outputReg, outputProb);
        size_t regOutputHeight = PNet_GetOutputRegHeight(outputHeight);
        size_t regOutputWidth = PNet_GetOutputRegWidth(outputWidth);

        size_t outputBoxMaxSize = 9*regOutputHeight*regOutputWidth;
//        size_t outputBoxMaxSize = 9*maxBoxesPerScale;
        float outputBox[outputBoxMaxSize];
        size_t boxesLen = MTCNN_GenerateBoundingBox(regOutputHeight, regOutputWidth, outputReg, outputProb, scales[i], thresholdPNet, outputBox);

//        if (i == 0){
//            for (size_t j=0;j<18;++j){
//                printf("Output [%d]: %f\n", j, outputBox[j]);
//                assert(equalFloatDefault(outputBox[j], expectedOutput01[j]));
//            }
//        }
//        else if (i == 1){
//            for (size_t j=0;j<18;++j){
//                printf("Output [%d]: %f\n", j, outputBox[j]);
//                assert(equalFloatDefault(outputBox[j], expectedOutput11[j]));
//            }
//        }
//        else if (i == 2){
//            for (size_t j=0;j<36;++j){
//                printf("Output [%d]: %f\n", j, outputBox[j]);
//                assert(equalFloatDefault(outputBox[j], expectedOutput21[j]));
//            }
//        }
//        else if (i == 3){
//            for (size_t j=0;j<36;++j){
//                printf("Output [%d]: %f\n", j, outputBox[j]);
//                assert(equalFloatDefault(outputBox[j], expectedOutput31[j]));
//            }
//        }
//        else if (i == 4){
//            for (size_t j=0;j<9;++j){
//                printf("Output [%d]: %f\n", j, outputBox[j]);
//                assert(equalFloatDefault(outputBox[j], expectedOutput41[j]));
//            }
//        }
        currentBoxesCount += MTCNN_BoxNms(boxesLen, outputBox, iouThresholdPNet0, boxes+currentBoxesCount*9);
//        free(scaledOutput);
    }
    if (currentBoxesCount == 0)
        return 0;

    currentBoxesCount = MTCNN_BoxNms(currentBoxesCount, boxes, iouThresholdPNet1, boxes);
//    float expectedOutput[] = {51.0, 21.0, 70.0, 40.0, 0.9039, -0.03787, 0.01272, 0.00483, 0.22522, 54.0, 28.0, 73.0, 46.0, 0.95487, -0.05535, 0.12994, -0.09492, 0.26756, 44.0, 25.0, 70.0, 51.0, 0.92829, 0.11208, 0.01449, 0.05255, 0.14467, 43.0, 29.0, 79.0, 66.0, 0.97301, 0.02569, -0.03714, -0.10525, 0.06338, 23.0, 23.0, 74.0, 74.0, 0.99966, 0.03534, -0.03008, -0.0268, 0.10018, 19.0, 19.0, 92.0, 92.0, 0.99322, 0.09031, 0.02829, -0.19878, 0.00652};
//    for (size_t i=0;i<54;++i){
//        printf("Output [%d]: %f\n", i, boxes[i]);
//        assert(equalFloatDefault(boxes[i], expectedOutput[i]));
//    }

    size_t outputIndex = 0;
    for (size_t i=0;i<currentBoxesCount*9;i+=9) {
        float regW = boxes[i+2] - boxes[i];
        float regH = boxes[i+3] - boxes[i+1];
        boxes[outputIndex] = boxes[i] + boxes[i+5] * regW;
        boxes[outputIndex+1] = boxes[i+1] + boxes[i+6] * regH;
        boxes[outputIndex+2] = boxes[i+2] + boxes[i+7] * regW;
        boxes[outputIndex+3] = boxes[i+3] + boxes[i+8] * regH;
        boxes[outputIndex+4] = boxes[i+4];
        outputIndex += 5;
    }
//    float expectedOutput[] = {50.28052, 21.24174, 70.09169, 44.27912, 0.9039, 52.94841, 30.33889, 71.19657, 50.81613, 0.95487,46.91397, 25.37681, 71.36622, 54.7613, 0.92829, 43.92469, 27.62577, 75.2111, 68.34497, 0.97301, 24.80256, 21.46568, 72.63345, 79.10918, 0.99966, 25.59233, 21.06518, 77.4894, 92.47623, 0.99322};
//    for (size_t i=0;i<30;++i){
//        printf("Output [%d]: %f\n", i, boxes[i]);
//        assert(equalFloatDefault(boxes[i], expectedOutput[i]));
//    }
    MTCNN_Rerec(currentBoxesCount, boxes);
    int padArray[currentBoxesCount*4];
    MTCNN_Pad(inputHeight, inputWidth, currentBoxesCount, boxes, padArray);

    float outputReg[currentBoxesCount*4];
    float outputProb[currentBoxesCount*2];
    size_t currentRegCount = 0;
    for (size_t i=0;i<currentBoxesCount*4;i+=4){
        size_t startH = padArray[i+1]-1;
        size_t stopH = padArray[i+3];
        size_t startW = padArray[i]-1;
        size_t stopW = padArray[i+2];
        if (stopH <= startH || stopW <= startW)
            continue;
        size_t newHeight = stopH-startH;
        size_t newWidth = stopW-startW;
        uint8_t newInput[inputChannels*newHeight*newWidth];
        for (size_t o=0;o<inputChannels;++o){
            for (size_t j=startH;j<stopH;++j){
                size_t inputIndex = o*inputWidth*inputHeight + inputWidth*j + startW;
                size_t newInputIndex = o*newWidth*newHeight + (j-startH)*newWidth;
                memcpy(newInput+newInputIndex, input+inputIndex, newWidth*sizeof(uint8_t));
            }
        }
//        if (i/4 == 0){
//            for (size_t k=0;k<1728;++k){
//                printf("Output [%d]: %f\n", k, newInput[k]);
//                assert(equalFloatDefault(newInput[k], expectedOutput02[k]));
//            }
//        }
//        else if (i/4 == 1){
//            for (size_t k=0;k<1386;++k){
//                printf("Output [%d]: %f\n", k, newInput[k]);
//                assert(equalFloatDefault(newInput[k], expectedOutput12[k]));
//            }
//        }
//        else if (i/4 == 2){
//            for (size_t k=0;k<2700;++k){
//                printf("Output [%d]: %f\n", k, newInput[k]);
//                assert(equalFloatDefault(newInput[k], expectedOutput22[k]));
//            }
//        }
//        else if (i/4 == 3){
//            for (size_t k=0;k<5166;++k){
//                printf("Output [%d]: %f\n", k, newInput[k]);
//                assert(equalFloatDefault(newInput[k], expectedOutput32[k]));
//            }
//        }
//        else if (i/4 == 4){
//            for (size_t k=0;k<10443;++k){
//                printf("Output [%d]: %f\n", k, newInput[k]);
//                assert(equalFloatDefault(newInput[k], expectedOutput42[k]));
//            }
//        }
//        else if (i/4 == 5){
//            for (size_t k=0;k<15768;++k){
//                printf("Output [%d]: %f\n", k, newInput[k]);
//                assert(equalFloatDefault(newInput[k], expectedOutput52[k]));
//            }
//        }
        float scaledInput[inputChannels*24*24];
        CNN_AdaptiveAveragePool_Uint8(inputChannels, newHeight, newWidth, 24, 24, newInput, scaledInput);
//        if (i/4 == 1){
//            for (size_t k=0;k<1728;++k){
//                printf("Output [%d]: %f\n", k, scaledInput[k]);
//                assert(equalFloatDefault(scaledInput[k], expectedOutput03[k]));
//            }
//        }
        for (size_t j=0;j<inputChannels*24*24;++j){
            scaledInput[j] = (scaledInput[j] - 127.5f) * 0.0078125f;
        }
        RNet_Model(scaledInput, outputReg+4*currentRegCount, outputProb+2*currentRegCount);
        ++currentRegCount;
    }
//    float expectedOutput04[] = {-0.12459, -0.0593, 0.22122, 0.73505, -0.11502, -0.12154, 0.21121, 0.48961, -0.05304, -0.05735, 0.13819, 0.47433, -0.18486, -0.1617, -0.14764, 0.19548,  0.03958, -0.02145, -0.11738, -0.0546, 0.1054, -0.01394, -0.18461, -0.1425};
//    for (size_t i=0;i<24;++i){
//        printf("Output [%d]: %f\n", i, outputReg[i]);
//        assert(equalFloatDefault(outputReg[i], expectedOutput04[i]));
//    }
//    float expectedOutput14[] = {0.99771, 0.00229, 0.99613, 0.00387, 0.99804, 0.00196, 0.09634, 0.90366, 0.00044, 0.99956, 0.00296, 0.99704};
//    for (size_t i=0;i<12;++i){
//        printf("Output [%d]: %f\n", i, outputProb[i]);
//        assert(equalFloatDefault(outputProb[i], expectedOutput14[i]));
//    }

    currentBoxesCount = 0;
    for(size_t i=0;i<currentRegCount;++i){
        size_t probIndex = 2*i+1;
        if (outputProb[probIndex] > thresholdRNet){
            boxes[5*currentBoxesCount] = boxes[5*i];
            boxes[5*currentBoxesCount+1] = boxes[5*i+1];
            boxes[5*currentBoxesCount+2] = boxes[5*i+2];
            boxes[5*currentBoxesCount+3] = boxes[5*i+3];
            boxes[5*currentBoxesCount+4] = outputProb[probIndex];

            outputReg[4*currentBoxesCount] = outputReg[4*i];
            outputReg[4*currentBoxesCount+1] = outputReg[4*i+1];
            outputReg[4*currentBoxesCount+2] = outputReg[4*i+2];
            outputReg[4*currentBoxesCount+3] = outputReg[4*i+3];
            ++currentBoxesCount;
        }
    }
//    float expectedOutput05[] = {39.20829, 27.62577, 79.92749, 68.34497, 0.90366, 19.89625, 21.46569, 77.53975, 79.10919, 0.99956, 15.83533, 21.06518, 87.24639, 92.47623, 0.99704};
//    for (size_t i=0;i<15;++i){
//        printf("Output [%d]: %f\n", i, boxes[i]);
//        assert(equalFloatDefault(boxes[i], expectedOutput05[i]));
//    }
//    float expectedOutput15[] = { -0.18486, -0.1617, -0.14764, 0.19548, 0.03958, -0.02145, -0.11738, -0.0546, 0.1054, -0.01394, -0.18461, -0.1425};
//    for (size_t i=0;i<12;++i){
//        printf("Output [%d]: %f\n", i, outputReg[i]);
//        assert(equalFloatDefault(outputReg[i], expectedOutput15[i]));
//    }
    int boxIndexes[currentBoxesCount];
    currentBoxesCount = MTCNN_BoxNmsIdx(currentBoxesCount, boxes, iouThresholdRNet, boxIndexes);
    for (size_t i=0;i<currentBoxesCount;++i){
        size_t inputIndex = boxIndexes[i];
        for (size_t j=0;j<5;++j){
            boxes[5*i+j] = boxes[5*inputIndex+j];
        }
        for (size_t j=0;j<4;++j){
            outputReg[4*i+j] = outputReg[4*inputIndex+j];
        }
    }
    for (size_t i=0;i<currentBoxesCount;++i) {
        size_t boxesIndex = 5*i;
        size_t regIndex = 4*i;
        float w = boxes[boxesIndex+2] - boxes[boxesIndex] + 1;
        float h = boxes[boxesIndex+3] - boxes[boxesIndex+1] + 1;
        boxes[boxesIndex] = boxes[boxesIndex] + outputReg[regIndex] * w;
        boxes[boxesIndex+1] = boxes[boxesIndex+1] + outputReg[regIndex+1] * h;
        boxes[boxesIndex+2] = boxes[boxesIndex+2] + outputReg[regIndex+2] * w;
        boxes[boxesIndex+3] = boxes[boxesIndex+3] + outputReg[regIndex+3] * h;
    }
//    float expectedOutput06[] = {22.21715, 20.20803, 70.65629, 75.90701, 0.99956};
//    for (size_t i=0;i<5;++i){
//        printf("Output [%d]: %f\n", i, boxes[i]);
//        assert(equalFloat(boxes[i], expectedOutput06[i], 0.01f));
//    }
    MTCNN_Rerec(currentBoxesCount, boxes);
    for (size_t i=0;i<currentBoxesCount*5;i+=5) {
        for (size_t j=0;j<5;++j){
            output[i+j] = boxes[i+j];
        }
    }
    return currentBoxesCount;
}

int MTCNN_GenerateBoundingBox(size_t inputHeight, size_t inputWidth, const float* reg, const float* score, float scale, float threshold, float* output){
    int stride = 2;
    int cellSize = 12;

    size_t maxIndexesSize = inputHeight * inputWidth;
    int indexes[maxIndexesSize];
    size_t idx = 0;
    for (size_t i=0;i<inputHeight;++i){
        for (size_t j=0;j<inputWidth;++j){
            size_t index = inputWidth*inputHeight + i*inputWidth + j; // equivalent to: score[1][i][j]
            if (score[index] > threshold){
//                if (idx >= MAX_INDEX_SIZE){
//                    break;
//                }
                indexes[idx] = index - inputWidth*inputHeight;
                ++idx;
            }
        }
    }
    float newReg[idx*4]; // same as above
    for (size_t i=0;i<idx;++i) {
        int row = indexes[i] / inputWidth;
        int column = indexes[i] % inputWidth;
        for (size_t o=0;o<4;++o){
            size_t index = o*inputHeight*inputWidth + row*inputWidth + column;
            newReg[4*i+o] = reg[index];
        }
    }

    float q1[idx*2];
    float q2[idx*2];
    for (size_t i=0;i<idx;++i) {
        int row = indexes[i] / inputWidth;
        int column = indexes[i] % inputWidth;
        q1[i*2] = floorf((stride * column + 1) / scale);
        q1[i*2+1] = floorf((stride * row + 1) / scale);
        q2[i*2] = floorf((stride * column + cellSize) / scale);
        q2[i*2+1] = floorf((stride * row + cellSize) / scale);
    }

    for (size_t i=0;i<idx;++i) {
        output[9*i] = q1[i*2];
        output[9*i+1] = q1[i*2+1];
        output[9*i+2] = q2[i*2];
        output[9*i+3] = q2[i*2+1];
        output[9*i+4] = score[indexes[i] + inputHeight*inputWidth]; // equivalent to: score[1][index]
        output[9*i+5] = newReg[4*i];
        output[9*i+6] = newReg[4*i+1];
        output[9*i+7] = newReg[4*i+2];
        output[9*i+8] = newReg[4*i+3];
    }
    return idx;
}

int MTCNN_BoxNms(size_t boxesLen, const float* boxes, float iouThreshold, float* output){
    if (boxesLen == 1){
        memcpy(output, boxes, 9*sizeof(float));
        return 1;
    }
    int indexes[boxesLen];
    for (size_t i=0;i<boxesLen;++i){
        indexes[i] = -1;
    }
    for (size_t i=0;i<boxesLen*9;i+=9){
        if (indexes[i/9] == -2)
            continue;
        for (size_t j=i+9;j<boxesLen*9;j+=9){
            if (indexes[j/9] == -2)
                continue;

            float iou = CNN_Iou(boxes[i], boxes[i+1], boxes[i+2], boxes[i+3], boxes[j], boxes[j+1], boxes[j+2], boxes[j+3]);
            int boxI = i;
            int boxJ = j;

            if (iou > iouThreshold){
                if (boxes[i+4] >= boxes[j+4]){
                    boxJ = -2;
                }
                else{
                    boxI = -2;
                }
            }
            indexes[i/9] = boxI;
            indexes[j/9] = boxJ;
            if (boxI == -2)
                break;
        }
    }
    size_t outputIndex = 0;
    int outputBoxesLen = 0;
    for (size_t i=0;i<boxesLen;++i){
        int inputIndex = indexes[i];
        if (inputIndex < 0)
            continue;
        output[outputIndex] = boxes[inputIndex];
        output[outputIndex+1] = boxes[inputIndex+1];
        output[outputIndex+2] = boxes[inputIndex+2];
        output[outputIndex+3] = boxes[inputIndex+3];
        output[outputIndex+4] = boxes[inputIndex+4];
        output[outputIndex+5] = boxes[inputIndex+5];
        output[outputIndex+6] = boxes[inputIndex+6];
        output[outputIndex+7] = boxes[inputIndex+7];
        output[outputIndex+8] = boxes[inputIndex+8];
        outputIndex += 9;
        ++outputBoxesLen;
    }
    return outputBoxesLen;
}

int MTCNN_BoxNmsIdx(size_t boxesLen, const float* boxes, float iouThreshold, int* boxesIndexes){
    int indexes[boxesLen];
    for (size_t i=0;i<boxesLen;++i){
        indexes[i] = -1;
    }
    for (size_t i=0;i<boxesLen*5;i+=5){
        if (indexes[i/5] == -2)
            continue;
        for (size_t j=i+5;j<boxesLen*5;j+=5){
            if (indexes[j/5] == -2)
                continue;

            float iou = CNN_Iou(boxes[i], boxes[i+1], boxes[i+2], boxes[i+3], boxes[j], boxes[j+1], boxes[j+2], boxes[j+3]);
            int boxI = i;
            int boxJ = j;

            if (iou > iouThreshold){
                if (boxes[i+4] >= boxes[j+4]){
                    boxJ = -2;
                }
                else{
                    boxI = -2;
                }
            }
            indexes[i/5] = boxI;
            indexes[j/5] = boxJ;
            if (boxI == -2)
                break;
        }
    }
    size_t outputIndex = 0;
    for (size_t i=0;i<boxesLen;++i){
        int inputIndex = indexes[i];
        if (inputIndex < 0)
            continue;
        boxesIndexes[outputIndex] = inputIndex/5;
        ++outputIndex;
    }
    return outputIndex;
}

void MTCNN_Rerec(size_t boxesLen, float* boxes){
    for (size_t i=0;i<boxesLen*5;i+=5){
        float w = boxes[i+2] - boxes[i];
        float h = boxes[i+3] - boxes[i+1];
        float l = fmaxf(h, w);
        boxes[i] = boxes[i] + w*0.5 - l*0.5;
        boxes[i+1] = boxes[i+1] + h*0.5 - l*0.5;
        boxes[i+2] = boxes[i] + l;
        boxes[i+3] = boxes[i+1] + l;
    }
}

void MTCNN_Pad(size_t inputHeight, size_t inputWidth, size_t boxesLen, float* boxes, int* output){
    size_t outputIndex = 0;
    for (size_t i=0;i<boxesLen*5;i+=5){
        output[outputIndex] = boxes[i] < 1 ? 1 : boxes[i];
        output[outputIndex+1] = boxes[i+1] < 1 ? 1 : boxes[i+1];
        output[outputIndex+2] = boxes[i+2] > inputWidth ? inputWidth : boxes[i+2];
        output[outputIndex+3] = boxes[i+3] > inputHeight ? inputHeight : boxes[i+3];
        outputIndex += 4;
    }
}

void MTCNN_Bbreg(size_t boxesLen, const float* reg, float* boxes){

}
