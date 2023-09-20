
cmake_minimum_required(VERSION 3.10)

project(myTensorrt)


add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(MMDEPLOY_BUILD_SDK ON)
option(MMDEPLOY_SHARED_LIBS "build shared libs" ON)
option(MMDEPLOY_BUILD_SDK "build MMDeploy SDK" ON)
option(MMDEPLOY_BUILD_SDK_MONOLITHIC "build single lib for SDK API" ON)
set(MMDEPLOY_TARGET_DEVICES "cuda;cpu" CACHE STRING "target devices to support")
set(MMDEPLOY_TARGET_BACKENDS "ort;trt" CACHE STRING "target inference engines to support")
set(MMDEPLOY_CODEBASES "all" CACHE STRING "select OpenMMLab codebases")

set(TENSORRT_DIR "/home/cqjtu/NVIDIA/TensorRT-8.5.3.1")
set(CUDNN_DIR "/usr/local/cuda")
set(pplcv_DIR "/home/cqjtu/Documents/ppl.cv/cuda-build/install/lib/cmake/ppl")

if (NOT CMAKE_BUILD_TYPE) # 编译类型
    set(CMAKE_BUILD_TYPE Release CACHE STRING "choose 'Release' as default build type" FORCE)
endif ()
if (MMDEPLOY_SHARED_LIBS)  #
    set(MMDEPLOY_LIB_TYPE SHARED)
else ()
    set(MMDEPLOY_LIB_TYPE STATIC)
endif ()
set(MMDEPLOY_TASKS "" CACHE INTERNAL "")  # 内部缓存变量CACHE INTERNAL
if (MMDEPLOY_UBSAN_ENABLE)  # 检测C/C++程序中的未定义行为
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=undefined>)
    add_link_options(-fsanitize=undefined)
endif ()
add_library(MMDeployStaticModules INTERFACE)  # 创建三个CMake目标
add_library(MMDeployDynamicModules INTERFACE)  # INTERFACE属性不包含库的实际源代码或二进制文件，
add_library(MMDeployLibs INTERFACE)  # 而是用于定义库的接口，它包括了其他目标在使用该库时需要了解的信息。

if ((cuda IN_LIST MMDEPLOY_TARGET_DEVICES) OR (trt IN_LIST MMDEPLOY_TARGET_BACKENDS))
    include(cmake/cuda.cmake NO_POLICY_SCOPE)
endif ()

include(${CMAKE_SOURCE_DIR}/cmake/MMDeploy.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tensorrt.cmake)
# Add block directories
add_subdirectory(tensorrt)

#add_subdirectory(csrc/mmdeploy)
if (MMDEPLOY_BUILD_SDK)
#    if (NOT MMDEPLOY_BUILD_SDK_MONOLITHIC)
#        install(TARGETS MMDeployStaticModules
#                MMDeployDynamicModules
#                MMDeployLibs
#                EXPORT MMDeployTargets)  # CMake目标集合 可以使用find_package命令来查找和引用
#    endif ()
#        # export MMDeploy package
#    install(EXPORT MMDeployTargets
#            FILE MMDeployTargets.cmake  # 导出的CMake配置文件的名称
#            DESTINATION lib/cmake/MMDeploy)  # 指定了将要安装配置文件的目标路径
    # append backend deps
    mmdeploy_add_deps(trt BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS TENSORRT CUDNN)
    mmdeploy_add_deps(ort BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS ONNXRUNTIME)
    if (NOT MMDEPLOY_SHARED_LIBS)
        mmdeploy_add_deps(pplnn BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS pplnn)
    endif ()
    include(CMakePackageConfigHelpers)

    install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/MMDeployConfig.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/MMDeployConfigVersion.cmake
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MMDeploy.cmake
            DESTINATION lib/cmake/MMDeploy
    )
    install(DIRECTORY
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules
            DESTINATION lib/cmake/MMDeploy
    )
endif ()
#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
#enable_language(CUDA)

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
        src/utils.h

)
# 引入算子
#link_directories(${PROJECT_SOURCE_DIR}/lib)

target_link_libraries(myTensorrt PRIVATE dl)
set_target_properties(myTensorrt PROPERTIES
        INSTALL_RPATH "cmake-build-debug/tensorrt"
)
#target_link_libraries(myTensorrt  PRIVATE mmdeploy_tensorrt_ops)
target_link_libraries(myTensorrt PRIVATE nvinfer nvparsers nvinfer_plugin nvonnxparser)
target_link_libraries(myTensorrt PRIVATE cudart)
target_link_libraries(myTensorrt PRIVATE ${OpenCV_LIBS} -lpthread -lm)

