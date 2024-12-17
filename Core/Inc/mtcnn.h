//
// Created by Michał on 15.11.2024.
//

#ifndef CNN_MTCNN_H
#define CNN_MTCNN_H

#include <stddef.h>
#include <stdint.h>

int MTCNN_DetectFace(size_t inputChannels, size_t inputHeight, size_t inputWidth, const uint8_t* input, float* output);

int MTCNN_GenerateBoundingBox(size_t inputHeight, size_t inputWidth, const float* reg, const float* score, float scale, float threshold, float* output);

int MTCNN_BoxNms(size_t boxesLen, const float* boxes, float iouThreshold, float* output);

int MTCNN_BoxNmsIdx(size_t boxesLen, const float* boxes, float iouThreshold, int* boxesIndexes);

void MTCNN_Rerec(size_t boxesLen, float* boxes);

void MTCNN_Pad(size_t inputHeight, size_t inputWidth, size_t boxesLen, float* boxes, int* output);

void MTCNN_Bbreg(size_t boxesLen, const float* reg, float* boxes);

#endif //CNN_MTCNN_H
