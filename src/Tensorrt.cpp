//
// Created by cqjtu on 23-9-14.
//

#include "Tensorrt.h"

TensorRT::TensorRT(common::InputParams inputParams, common::TrtParams trtParams)
        : mInputParams(std::move(inputParams)), mTrtParams(std::move(trtParams)) ,
          mThreadPool(new tss::thread_pool(trtParams.worker)){
    CHECK(cudaEventCreate(&this->start_t));
    CHECK(cudaEventCreate(&this->stop_t));
}
bool TensorRT::constructNetwork(TensorRT::UniquePtr<nvinfer1::IBuilder> &builder,
                                TensorRT::UniquePtr<nvinfer1::INetworkDefinition> &network,
                                TensorRT::UniquePtr<nvinfer1::IBuilderConfig> &config,
                                TensorRT::UniquePtr<nvonnxparser::IParser> &parser,
                                const std::string &onnxPath) const {
    std::cout<<"hallo parsed:"<<onnxPath.c_str()<<std::endl;
    auto parsed = parser->parseFromFile(onnxPath.c_str(),
                                        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        gLogError << "Failure while parsing ONNX file" << std::endl;
        for (int32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cout << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }

    if (mTrtParams.FP16)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (mTrtParams.Int8)
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        common::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }

    common::enableDLA(builder.get(), config.get(), mTrtParams.useDLA);

    return true;
}
bool TensorRT::buildToengine() {
    initLibNvInferPlugins(&gLogger, "");
    std::cout<<"hallo build"<<std::endl;
    auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) {
        gLogError << "Create Builder Failed" << std::endl;
        return false;
    }
    std::cout<<"hallo network"<<std::endl;
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = UniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        gLogError << "Create Network Failed" << std::endl;
        return false;
    }
    //优化配置
    std::cout<<"hallo config"<<std::endl;
    auto config = UniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        gLogError << "Create Config Failed" << std::endl;
        return false;
    }
    //ONNX 解析器
    std::cout<<"hallo parser"<<std::endl;
    auto parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser) {
        gLogError << "Create Parser Failed" << std::endl;
        return false;
    }

//////sampleOnnxMNIST
    std::cout<<"hallo constructed"<<std::endl;
    auto constructed = constructNetwork(builder, network, config, parser, mTrtParams.OnnxPath);
    if (!constructed)
    {
        return false;
    }
    // CUDA stream used for profiling by the builder.
    std::cout<<"hallo profileStream"<<std::endl;
    auto profileStream = common::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);
    UniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }
    UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime)
    {
        gLogError << "Create runtime Failed" << std::endl;
        return false;
    }
    mCudaEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(plan->data(), plan->size()), common::InferDeleter());
    if (!mCudaEngine)
    {gLogError << "Create Engine Failed" << std::endl;
        return false;
    }

    ASSERT(network->getNbInputs() ==mInputParams.InputTensorNames.size());
    mInputParams.mInputDims = network->getInput(0)->getDimensions();
    std::cout<<"mInputDims:"<<mInputParams.mInputDims.nbDims<<std::endl;
    ASSERT(mInputParams.mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() ==mInputParams.OutputTensorNames.size());
    mInputParams.mOutputDims = network->getOutput(0)->getDimensions();
    std::cout<<"mInputDims:"<<mInputParams.mOutputDims.nbDims<<std::endl;
    ASSERT(mInputParams.mOutputDims.nbDims == 4);
    ///////

    mContext = UniquePtr<nvinfer1::IExecutionContext>(mCudaEngine->createExecutionContext());
    if(!mContext){
        gLogError << "Create Context Failed" << std::endl;
        return false;
    }

    return true;
}

bool TensorRT::initSession(int initOrder) {
    CHECK(cudaSetDevice(0));
    loadLibrary(std::string("/home/cqjtu/CLionProjects/myTensorrt/lib/libmmdeploy_tensorrt_ops.so"));
    if(initOrder==0){
        if(!this->deseriazeEngine(mTrtParams.SerializedPath)){
            this->buildToengine();
        }
    } else if(initOrder==1){
        if(!this->buildToengine()){
            gLogError << "Init Session Failed!" << std::endl;
            return false;
        }
    }


    return true;
}

TensorRT::~TensorRT() {

}
bool TensorRT::serializeEngine(const std::string &save_path) {
    nvinfer1::IHostMemory *gie_model_stream = mCudaEngine -> serialize();
    std::ofstream serialize_output_stream;
    std::string serialize_str;
    serialize_str.resize(gie_model_stream->size());
    memcpy((void*)serialize_str.data(),gie_model_stream->data(), gie_model_stream->size());
    serialize_output_stream.open(save_path);
    if(!serialize_output_stream.good()){
        gLogError << "Serializing Engine Failed" << std::endl;
        return false;
    }
    serialize_output_stream<<serialize_str;
    serialize_output_stream.close();
    return true;
}
using namespace std;
bool TensorRT::deseriazeEngine(const std::string &enginePath) {
    gLogInfo << "--------------------\n";

    initLibNvInferPlugins(&gLogger, "");

    gLogInfo << "Loading BERT Inference Engine ... \n";
    std::ifstream input(enginePath, std::ios::binary);
    if (!input)
    {
        gLogError << "Error opening engine file: " << enginePath << "\n";
        exit(-1);
    }

    input.seekg(0, input.end);
    const size_t fsize = input.tellg();
    input.seekg(0, input.beg);

    std::vector<char> bytes(fsize);
    input.read(bytes.data(), fsize);

    UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime)
    {
        gLogError << "Create runtime Failed" << std::endl;
        return false;
    }
    initLibNvInferPlugins(&gLogger, "");

    mCudaEngine = UniquePtr<nvinfer1::ICudaEngine>( runtime->deserializeCudaEngine(bytes.data(),
                                                                                   bytes.size()), common::InferDeleter());
    if (!mCudaEngine)
    {gLogError << "Create Engine Failed" << std::endl;
        return false;
    }
    assert(mCudaEngine != nullptr);
    vector<char>().swap(bytes);
    gLogInfo << "Succeeded loading engine!\n";

#ifdef TEST
    // 获取模型的输入数量
    int numInputs = mCudaEngine->getNbBindings() / 2; // 每个输入有两个绑定（数据和维度）
    // 打印模型的输入维度
    for (int i = 0; i < numInputs; ++i) {
        const char* inputName = mCudaEngine->getBindingName(i * 2); // 输入的数据绑定
        nvinfer1::Dims inputDims = mCudaEngine->getBindingDimensions(i * 2);
        std::cout << "Input " << i << " Name: " << inputName << ", Dimensions: ";
        for (int j = 0; j < inputDims.nbDims; ++j) {
            std::cout << inputDims.d[j] << " ";
        }
        std::cout << std::endl;
    }
#endif

    return true;
}



float TensorRT::infer(const std::vector<std::vector<float>> &InputDatas, common::BufferManager &bufferManager,
                      cudaStream_t stream) const {

    assert(InputDatas.size()==mInputParams.InputTensorNames.size());
    for(int i=0; i<InputDatas.size(); ++i){
        std::memcpy((void*)bufferManager.getHostBuffer(mInputParams.InputTensorNames[i]), (void*)InputDatas[i].data(), InputDatas[i].size() * sizeof(float));
    }
    bufferManager.copyInputToDeviceAsync();
    CHECK(cudaEventRecord(this->start_t, stream));
    if (!mContext->enqueueV2(bufferManager.getDeviceBindings().data(), stream, nullptr)) {
        gLogError << "Execute Failed!" << std::endl;
        return false;
    }
    CHECK(cudaEventRecord(this->stop_t, stream));
    bufferManager.copyOutputToHostAsync();
    float elapsed_time;
    CHECK(cudaEventSynchronize(this->stop_t));
    CHECK(cudaEventElapsedTime(&elapsed_time, this->start_t, this->stop_t));
    return elapsed_time;
}

bool TensorRT::LoadEngine() {
    CHECK(cudaSetDevice(0));
    loadLibrary(std::string("/home/cqjtu/CLionProjects/myTensorrt/lib/libmmdeploy_tensorrt_ops.so"));

// create and load engine
    try {this->deseriazeEngine(mTrtParams.SerializedPath);}
    catch (const std::exception& e){
        std::cerr << "Exception: " << e.what() << std::endl;
        this->OnnxToTRTModel(mTrtParams.SerializedPath);
    }
    mContext = UniquePtr<nvinfer1::IExecutionContext>(mCudaEngine->createExecutionContext());
    if(!mContext){
        gLogError << "Create Context Failed" << std::endl;
        return false;
    }
    gLogError << "Succeeded Create Context" << std::endl;

    assert(mCudaEngine->getNbIOTensors() == 4);
    int nbBindings = mCudaEngine->getNbIOTensors();
    assert(mInputParams.BatchSize==1);

/*    assert(engine->getNbBindings() == 2);
//    int nbBindings = engine->getNbBindings();
//    bufferSize.resize(nbBindings);
//    for (int i = 0; i < nbBindings; ++i) {
//        nvinfer1::Dims dims = engine->getBindingDimensions(i);
//        nvinfer1::DataType dtype = engine->getBindingDataType(i);
//        int64_t totalSize = volume(dims) * getElementSize(dtype);
//        bufferSize[i] = totalSize;
//        std::cout << "binding" << i << ": " << totalSize << std::endl;
//        cudaMalloc(&buffers[i], totalSize);}
*/

}

bool TensorRT::OnnxToTRTModel(const std::string& engine_file) {
    this->buildToengine();
    this->serializeEngine(engine_file);

    return true;
}

bool TensorRT::allocator() {

    return false;
}

std::vector<common::Bbox> TensorRT::predOneImage(const cv::Mat &image, float postThres, float nmsThres) {
    ///prepare data
    assert(mInputParams.BatchSize==1);
    common::BufferManager bufferManager(mCudaEngine, 1);

    Clock<std::chrono::high_resolution_clock > clock_t;
    clock_t.tick();
    auto preImg = preProcess(std::vector<cv::Mat>{image});
    clock_t.tock();
    gLogInfo << "Pre Process time is " << clock_t.duration<double>() << "ms"<< std::endl;
    ///infer
    float elapsedTime = this->infer(std::vector<std::vector<float>>{preImg}, bufferManager, nullptr);
    gLogInfo << "Infer time is "<< elapsedTime << "ms" << std::endl;

    ///post progress
//    clock_t.tick();
//    cv::Mat mask = postProcess(bufferManager, postThres);
//    clock_t.tock();
//    gLogInfo << "Post Process time is " << clock_t.duration<double>() << "ms"<< std::endl;
//    this->transform(image.rows, image.cols, mInputParams.ImgH, mInputParams.ImgW, mask, mInputParams.IsPadding);
//    return mask;
    return std::vector<common::Bbox>();
}

