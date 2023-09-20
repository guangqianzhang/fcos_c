#include <iostream>
#include "Tensorrt.h"
void initInputParams(common::InputParams &inputParams){
    inputParams.ImgH = 1600;
    inputParams.ImgW = 900;
    inputParams.ImgC = 3;
    inputParams.BatchSize = 1;
    inputParams.IsPadding = true;
    inputParams.InputTensorNames = std::vector<std::string>{"img","cam2img_inerse"};
    inputParams.OutputTensorNames = std::vector<std::string>{"bboxes","dir_scores"};
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
    trt.LoadEngine();
//using namespace std;
//    char *trtModelStream{nullptr};
//    const std::string engine_file_path =trtParams.OnnxPath;
//    std::ifstream file(engine_file_path, std::ios::binary);
//    if (file.good()) {
//        file.seekg(0, file.end);
//        size_t size = file.tellg();
//        file.seekg(0, file.beg);
//        trtModelStream = new char[size];
//        assert(trtModelStream);
//        file.read(trtModelStream, size);
//        file.close();
//    }


//    std::ifstream file(trtParams.SerializedPath, std::ios::binary);
//    if (!file.good()) {
//        std::cerr << "read " << trtParams.SerializedPath << " error!" << std::endl;
//        assert(false);
//    }
//    std::cout << "loading filename from:" << trtParams.SerializedPath << std::endl;
//    size_t size = 0;
//    file.seekg(0, file.end);
//    size = file.tellg();
//    file.seekg(0, file.beg);
//    char *serialized_engine = new char[size];
//    assert(serialized_engine);
//    file.read(serialized_engine, size);
//    file.close();
//std::cout<<trtModelStream<<std::endl;
    return 0;
}
