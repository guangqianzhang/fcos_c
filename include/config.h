//
// Created by zgq on 23-9-5.
//

#ifndef FCOS_C_CONFIG_H
#define FCOS_C_CONFIG_H

#define USE_FP16
//#define USE_INT8

const static char *kInputTensorName = "images";
const static char *kOutputTensorName = "output";
const static int kNumClass = 80;
const static int kBatchSize = 1;
const static int kGpuId = 0;
const static int kInputH = 640;
const static int kInputW = 640;
const static float kNmsThresh = 0.45f;
const static float kConfThresh = 0.5f;
const static int kMaxInputImageSize = 3000 * 3000;
// Maximum number of output bounding boxes from yololayer plugin.
// That is maximum number of output bounding boxes before NMS.
constexpr static int kMaxNumOutputBbox = 1000;

#endif //FCOS_C_CONFIG_H
