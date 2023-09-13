//
// Created by cqjtu on 23-9-13.
//

#ifndef FCOS_CUDA_UTILS_HPP
#define FCOS_CUDA_UTILS_HPP

#include <iostream>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);

#endif //FCOS_CUDA_UTILS_HPP
