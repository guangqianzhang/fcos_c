//
// Created by zgq on 23-9-5.
//

#ifndef FCOS_C_COMMON_H
#define FCOS_C_COMMON_H
#include <NvInfer.h>
#include <vector>
#include <string>
#include <iostream>
#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include "logger.h"
#include <chrono>
#include <utils.h>
#define CHECK( err ) (HandleError( err, __FILE__, __LINE__ ))
static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

namespace common{
    // <============== Params =============>
    struct InputParams {
        // General
        int ImgC;
        int ImgH;
        int ImgW;
        int BatchSize;
        bool IsPadding;
        bool HWC;
        // Tensor
        std::vector<std::string> InputTensorNames;
        std::vector<std::string> OutputTensorNames;
        // Image pre-process function
        float(*pFunction)(const unsigned char&);
        nvinfer1::Dims mInputDims;
        nvinfer1::Dims mOutputDims;
        InputParams() : ImgC(0), ImgH(0), ImgW(0), BatchSize(0), IsPadding(true), HWC(false), InputTensorNames(),
                        OutputTensorNames(), pFunction(nullptr){
        };
    };

    struct TrtParams{
        std::size_t ExtraWorkSpace;
        bool FP32;
        bool FP16;
        bool Int32;
        bool Int8;
        int useDLA;
        int worker;
        int MaxBatch;
        int MinTimingIteration;
        int AvgTimingIteration;
        std::string CalibrationTablePath;
        std::string CalibrationImageDir;
        std::string OnnxPath;
        std::string SerializedPath;
        TrtParams() : ExtraWorkSpace(0), FP32(true), FP16(false), Int32(false), Int8(false), useDLA(-1), worker(0),
                      MaxBatch(100), MinTimingIteration(1), AvgTimingIteration(2), CalibrationImageDir(), CalibrationTablePath(),
                      OnnxPath(), SerializedPath(){
        };
    };

    struct Anchor{
        float width;
        float height;
        Anchor() : width(0), height(0){
        };
    };

    struct DetectParams{
        // Detection/SegmentationTRT
        std::vector<int> Strides;
        std::vector<common::Anchor> Anchors;
        int AnchorPerScale;
        int NumClass;
        float NMSThreshold;
        float PostThreshold;
        DetectParams() : Strides(), Anchors(), AnchorPerScale(0), NumClass(0), NMSThreshold(0), PostThreshold(0) {
        };
    };

    struct KeypointParams{
        // Hourglass
        int HeatMapH;
        int HeatMapW;
        int NumClass;
        float PostThreshold;
        KeypointParams() : HeatMapH(0), HeatMapW(0), NumClass(0), PostThreshold(0) {
        };
    };

    struct ClassificationParams{
        int NumClass;
        ClassificationParams() : NumClass(0){
        };
    };

    // <============== Outputs =============>
    struct Bbox{
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        float score;
        int cid;
        Bbox() : xmin(0), ymin(0), xmax(0), ymax(0), score(0), cid(0) {
        };
    };

    struct Keypoint{
        float x;
        float y;
        float score;
        int cid;
        Keypoint() : x(0), y(0), score(0), cid(0) {
        }
    };


    // <============== Operator =============>
    struct InferDeleter{
        template <typename T>
        void operator()(T* obj) const
        {
            if (obj)
            {
                obj->destroy();
            }
        }
    };
#undef ASSERT
#define ASSERT(condition)                                                   \
    do                                                                      \
    {                                                                       \
        if (!(condition))                                                   \
        {                                                                   \
            gLogError << "Assertion failure: " << #condition << std::endl;  \
            abort();                                                        \
        }                                                                   \
    } while (0)


    //<============DLA=================>
    inline void enableDLA(
            nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true)
    {
        if (useDLACore >= 0)
        {
            if (builder->getNbDLACores() == 0)
            {
                std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
                          << std::endl;
                assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
            }
            if (allowGPUFallback)
            {
                config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
            }
            if (!config->getFlag(nvinfer1::BuilderFlag::kINT8))
            {
                // User has not requested INT8 Mode.
                // By default run in FP16 mode. FP32 mode is not permitted.
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
            }
            config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            config->setDLACore(useDLACore);
        }
    }
    inline void setAllDynamicRanges(nvinfer1::INetworkDefinition* network, float inRange = 2.0f, float outRange = 4.0f)
    {
        // Ensure that all layer inputs have a scale.
        for (int i = 0; i < network->getNbLayers(); i++)
        {
            auto layer = network->getLayer(i);
            for (int j = 0; j < layer->getNbInputs(); j++)
            {
                nvinfer1::ITensor* input{layer->getInput(j)};
                // Optional inputs are nullptr here and are from RNN layers.
                if (input != nullptr && !input->dynamicRangeIsSet())
                {
                    ASSERT(input->setDynamicRange(-inRange, inRange));
                }
            }
        }

        // Ensure that all layer outputs have a scale.
        // Tensors that are also inputs to layers are ingored here
        // since the previous loop nest assigned scales to them.
        for (int i = 0; i < network->getNbLayers(); i++)
        {
            auto layer = network->getLayer(i);
            for (int j = 0; j < layer->getNbOutputs(); j++)
            {
                nvinfer1::ITensor* output{layer->getOutput(j)};
                // Optional outputs are nullptr here and are from RNN layers.
                if (output != nullptr && !output->dynamicRangeIsSet())
                {
                    // Pooling must have the same input and output scales.
                    if (layer->getType() == nvinfer1::LayerType::kPOOLING)
                    {
                        ASSERT(output->setDynamicRange(-inRange, inRange));
                    }
                    else
                    {
                        ASSERT(output->setDynamicRange(-outRange, outRange));
                    }
                }
            }
        }
    }
    static auto StreamDeleter = [](cudaStream_t* pStream)
    {
        if (pStream)
        {
            cudaStreamDestroy(*pStream);
            delete pStream;
        }
    };
    inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
    {
        std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
        if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
        {
            pStream.reset(nullptr);
        }

        return pStream;
    }
}


#endif //FCOS_C_COMMON_H
