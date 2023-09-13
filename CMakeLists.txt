cmake_minimum_required(VERSION 3.26)
project(fcos)
enable_language(CUDA)#激活CUDA语言支持，使用第一个写法时要进行注释
set(CMAKE_CXX_STANDARD 17)

## cuda
add_compile_options(-g)
set(CMAKE_CUDA_FLAGS "-Wall")
set(CMAKE_CUDA_STANDARD 11)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")# x86
    # cuda
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)

    ## tensorrt
    include_directories(/home/cqjtu/NVIDIA/TensorRT-8.5.3.1/include)
    link_directories(/home/cqjtu/NVIDIA/TensorRT-8.5.3.1/lib)
endif()
#opencv
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "include:${OpenCV_INCLUDE_DIRS}")

#添加项目自身的库和依赖
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/cmake-build-debug)#设置编译后的程序存放位置
include_directories(${PROJECT_SOURCE_DIR}/include)#添加头文件路径
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp )


add_executable(fcos main.cpp ${SRCS}
        include/cuda_utils.hpp
        src/cuda_utils.cpp
        src/affine.cu
)
target_link_libraries(fcos  ${OpenCV_LIBS}
                            cudart nvinfer
                            )
