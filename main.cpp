#include <iostream>
#include "Tensorrt.h"
void initInputParams(common::InputParams &inputParams){
    inputParams.ImgH = 1600;
    inputParams.ImgW = 900;
    inputParams.ImgC = 3;
    inputParams.BatchSize = 1;
    inputParams.IsPadding = true;
    inputParams.InputTensorNames = std::vector<std::string>{"img","cam2img","cam2img_inerse"};
    inputParams.OutputTensorNames = std::vector<std::string>{"bboxes","score","labels","dir_scores","attrs"};
//    inputParams.pFunction = [](const unsigned char &x){return static_cast<float>(x) /255;};
}

void initTrtParams(common::TrtParams &trtParams){
    trtParams.ExtraWorkSpace = 0;
    trtParams.FP32 = true;
    trtParams.FP16 = false;
    trtParams.Int32 = false;
    trtParams.Int8 = false;
    trtParams.MaxBatch = 100;
    trtParams.MinTimingIteration = 1;
    trtParams.AvgTimingIteration = 2;
    trtParams.CalibrationTablePath = "/work/tensorRT-7/data/fcosInt8.calibration";
    trtParams.CalibrationImageDir = "/home/cqjtu/Documents/dataset/test";
    trtParams.OnnxPath = "/home/cqjtu/CLionProjects/myTensorrt/r101/end2end.onnx";
    trtParams.SerializedPath = "/home/cqjtu/CLionProjects/myTensorrt/r101/end2end.engine";
}

void initDetectParams(common::DetectParams &fcosParams){
    fcosParams.Strides = std::vector<int> {8, 16, 32, 64, 128};
    fcosParams.AnchorPerScale = -1;
    fcosParams.NumClass = 80;
    fcosParams.NMSThreshold = 0.45;
    fcosParams.PostThreshold = 0.3;
}
int main() {
    std::cout << "Hello, World!" << std::endl;
    common::InputParams inputParams;
    common::TrtParams trtParams;
    common::DetectParams fcosParams;
    initInputParams(inputParams);
    initTrtParams(trtParams);
    initDetectParams(fcosParams);
    TensorRT trt(inputParams, trtParams);
    trt.initSession(0);

    return 0;
}
