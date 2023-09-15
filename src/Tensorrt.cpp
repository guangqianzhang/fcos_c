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

bool TensorRT::buildengine() {
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
    ASSERT(mInputParams.mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() ==mInputParams.OutputTensorNames.size());
    mInputParams.mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mInputParams.mOutputDims.nbDims == 2);
    ///////

    mContext = UniquePtr<nvinfer1::IExecutionContext>(mCudaEngine->createExecutionContext());
    if(!mContext){
        gLogError << "Create Context Failed" << std::endl;
        return false;
    }

    return true;
}

bool TensorRT::initSession(int initOrder) {
    if(initOrder==0){
        if(!this->deseriazeEngine(mTrtParams.SerializedPath)){
            this->buildengine();
        }
    } else if(initOrder==1){
        if(!this->buildengine()){
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

bool TensorRT::deseriazeEngine(const std::string &load_path) {
    std::ifstream fin(load_path);
    if (!fin.good()){
        return false;
    }
    std::cout << "loading filename from:" << load_path << std::endl;
    std::string deserialize_str;
    while (fin.peek() != EOF){ // 使用fin.peek()防止文件读取时无限循环
        std::stringstream buffer;
        buffer << fin.rdbuf();
        deserialize_str.append(buffer.str());
    }
    fin.close();

    UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime)
    {
        gLogError << "Create runtime Failed" << std::endl;
        return false;
    }
    initLibNvInferPlugins(&gLogger, "");
    mCudaEngine = UniquePtr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(deserialize_str.data(),
                                           deserialize_str.size()), common::InferDeleter());
    if (!mCudaEngine)
    {gLogError << "Create Engine Failed" << std::endl;
        return false;
    }
    assert(mCudaEngine != nullptr);
    return true;
}

bool TensorRT::constructNetwork(TensorRT::UniquePtr<nvinfer1::IBuilder> &builder,
                                TensorRT::UniquePtr<nvinfer1::INetworkDefinition> &network,
                                TensorRT::UniquePtr<nvinfer1::IBuilderConfig> &config,
                                TensorRT::UniquePtr<nvonnxparser::IParser> &parser,
                                const std::string &onnxPath
                                ) const {
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

float TensorRT::infer(const std::vector<std::vector<float>> &InputDatas, common::BufferManager &bufferManager,
                      cudaStream_t stream) const {
    // Create RAII buffer manager object
    common::BufferManager buffers(mCudaEngine,1);

    auto context = UniquePtr<nvinfer1::IExecutionContext>(mCudaEngine->createExecutionContext());
    if (!context)
    {
        gLogError << "Init context Failed!" << std::endl;
        return false;
    }

    // Read the input data into the managed buffers
    ASSERT(mInputParams.InputTensorNames.size() == 1);
//    if (!processInput(buffers))
//    {
//        return false;
//    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results



    return 0;
}

bool TensorRT::LoadEngine() {
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
}

bool TensorRT::OnnxToTRTModel(const std::string& engine_file) {
    this->buildengine();
    this->serializeEngine(engine_file);

    return true;
}

