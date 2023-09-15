cmake_minimum_required(VERSION 3.26)
project(myTensorrt)


cmake_minimum_required(VERSION 3.10)


add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    # cuda
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)

    # tensorrt
    include_directories(/home/cqjtu/NVIDIA/TensorRT-8.5.3.1/include)
    link_directories(/home/cqjtu/NVIDIA/TensorRT-8.5.3.1/lib
            /home/cqjtu/NVIDIA/TensorRT-8.5.3.1/targets/x86_64-linux-gnu/lib)
    #  include_directories(/home/lindsay/TensorRT-7.2.3.4/include)
    #  link_directories(/home/lindsay/TensorRT-7.2.3.4/lib)


endif()

#add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
#target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_BINARY_DIR}/include ${PROJECT_SOURCE_DIR}/src)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
message("src: " ${SRCS})
add_executable(myTensorrt ${PROJECT_SOURCE_DIR}/main.cpp ${SRCS}

)
# 引入算子
#link_directories(${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(myTensorrt  mmdeploy_tensorrt_ops)

target_link_libraries(myTensorrt nvinfer nvparsers nvinfer_plugin nvonnxparser)
target_link_libraries(myTensorrt cudart)
target_link_libraries(myTensorrt ${OpenCV_LIBS} -lpthread -lm)

