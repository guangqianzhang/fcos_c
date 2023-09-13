//
// Created by cqjtu on 23-9-13.
//
#include "cuda_utils.hpp"

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){
        const char* err_name = cudaGetErrorName(code);
        const char* err_message = cudaGetErrorString(code);
        std::cerr<<"runtime error "<<file<<":"<<line<<" "<<op<<"failed"<<std::endl;
        std::cerr<<"code = "<<err_name<<" message = "<<err_message<<std::endl;
//        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}