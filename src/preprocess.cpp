//
// Created by zgq on 23-9-6.
//

#include "preprocess.h"
/*
 * 中心对齐的图片放缩、填充、归一化（0，1）
 * 实现方式cuda 放射变换
 * */
float* warpaffine_to_center_align(const cv::Mat& image, const cv::Size& size){
    float* output = new float[size.height*size.width*3];

    uint8_t* psrc_device = nullptr;
    float* pdst_device = nullptr;
    size_t src_size = image.cols * image.rows * 3;
    size_t dst_size = size.width * size.height * 3;

    checkRuntime(cudaMalloc(&psrc_device, src_size*sizeof(uint8_t))); // 在GPU上开辟两块空间
    checkRuntime(cudaMalloc(&pdst_device, dst_size*sizeof(float)));
    checkRuntime(cudaMemcpy(psrc_device, image.data, src_size, cudaMemcpyHostToDevice)); // 搬运数据到GPU上
    auto start_time = std::chrono::high_resolution_clock::now();
    warp_affine_bilinear(
            psrc_device, image.cols * 3, image.cols, image.rows,
            pdst_device, size.width * 3, size.width, size.height,
            114
    );
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "cost: " << duration.count() << " microseconds" << std::endl;
    // 检查核函数执行是否存在错误
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaMemcpy(output, pdst_device, dst_size*sizeof(float), cudaMemcpyDeviceToHost)); // 将预处理完的数据搬运回来
    checkRuntime(cudaFree(psrc_device));
    checkRuntime(cudaFree(pdst_device));
    return output;
}