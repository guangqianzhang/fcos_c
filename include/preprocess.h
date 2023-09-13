//
// Created by zgq on 23-9-5.
//

#ifndef FCOS_C_PREPROCESS_H
#define FCOS_C_PREPROCESS_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "types.h"
#include <map>
#include "cuda_utils.hpp"
#include "affine.cuh"

void cuda_preprocess_init(int max_image_size);

void cuda_preprocess_destroy();

void cuda_preprocess(uint8_t *src, int src_width, int src_height, uint8_t *dst, int dst_width, int dst_height, cudaStream_t stream);

void cuda_batch_preprocess(std::vector<cv::Mat> &img_batch, uint8_t *dst, int dst_width, int dst_height, cudaStream_t stream);

float* warpaffine_to_center_align(const cv::Mat& image, const cv::Size& size);


#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
#endif //FCOS_C_PREPROCESS_H
