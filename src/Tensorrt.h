//
// Created by zgq on 23-8-25.
//

#ifndef FCOS_C_TENSORRT_H
#define FCOS_C_TENSORRT_H
#include "common.h"
#include "buffers.h"
#include "logger.h"
//#include "Int8Calibrator.h"
//#include "utils.h"
#include "thread_safety_stl.h"

#include <fstream>
#include <opencv2/core.hpp>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <memory>
#include <map>
#include <mutex>
#include <thread>
//! TensorRT Base Class
//#include "trt_modulated_deform_conv.hpp"

//REGISTER_TENSORRT_PLUGIN(AddScalarPluginCreator);
class TensorRT{
protected:
    template <typename T>
    using UniquePtr = std::unique_ptr<T, common::InferDeleter>;//删除器，用于在释放指针时执行特定的操作
    std::shared_ptr<nvinfer1::ICudaEngine> mCudaEngine;
    std::shared_ptr<nvinfer1::IExecutionContext> mContext;
    std::shared_ptr<tss::thread_pool> mThreadPool;
    common::InputParams mInputParams;
    common::TrtParams mTrtParams;
    cudaEvent_t start_t{}, stop_t{};

private:
    //! Build the network and construct CudaEngine and Context. It should be called before serializeEngine()
    //! \param onnxPath Onnx file path.
    //! \return Return true if no errors happened.
    bool buildToengine();

    //! Serialize the CudaEngine. It should be called after constructNetwork();
    //! \param save_path Saved file path
    //! \return Return true if no errors happened.
    bool serializeEngine(const std::string &save_path);

    //! Deserialize the CudaEngine.
    //! \param load_path Saved file path
    //! \return Return true if no errors happened.
    bool deseriazeEngine(const std::string &load_path);

    bool allocator();

//protected:
public:
    //! Initialize mInputParams, mTrtParms
    //! \param inputParams Input images params
    //! \param trtParams TensorRT definition configs
    TensorRT(common::InputParams inputParams, common::TrtParams trtParams);

    ~TensorRT();


    std::vector<common::Bbox> predOneImage(const cv::Mat &image, float postThres, float nmsThres) ;

    //! PreProcess
     std::vector<float> preProcess(const std::vector<cv::Mat> &images);

    //! (ASynchronously) Execute the inference on a batch.
    //! \param InputDatas Float arrays which must corresponded with InputTensorNames.
    //! \param bufferManager An instance of BufferManager. It holds the raw inference results.
    //! \param cudaStream_t Execute in stream
    //! \return Return the inference time in ms. If failed, return 0.
     float infer(const std::vector<std::vector<float>>&InputDatas, common::BufferManager &bufferManager, cudaStream_t stream) const;

//protected:
    //! Init Inference Session
    //! \param initOrder 0==========> init from SerializedPath. If failed, init from onnxPath.
    //!                             1 ==========> init from onnxPath and save the session into SerializedPath if it doesnt exist.
    //!                             2 ==========> init from onnxPath and force to save the session into SerializedPath.
    //! \return true if no errors happened.
    bool initSession(int initOrder);

    bool constructNetwork(UniquePtr<nvinfer1::IBuilder>& builder,
                          UniquePtr<nvinfer1::INetworkDefinition>& network, UniquePtr<nvinfer1::IBuilderConfig>& config,
                          UniquePtr<nvonnxparser::IParser>& parser,const std::string &onnxPath) const;
    bool LoadEngine();
    bool OnnxToTRTModel(const std::string& engine_file);
};
#include <dlfcn.h>
inline void loadLibrary(const std::string &path)
{
#ifdef _MSC_VER
    void *handle = LoadLibrary(path.c_str());
#else
    int32_t flags {RTLD_LAZY};
    void   *handle = dlopen(path.c_str(), flags);
#endif
    if (handle == nullptr)
    {
#ifdef _MSC_VER
        std::cout << "Could not load plugin library: " << path << std::endl;
#else
        std::cout << "Could not load plugin library: " << path << ", due to: " << dlerror() << std::endl;
#endif
    }
}


#endif //FCOS_C_TENSORRT_H
