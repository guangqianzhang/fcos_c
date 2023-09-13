//
// Created by zgq on 23-9-6.
//

#ifndef FCOS_AFFINE_CUH
#define FCOS_AFFINE_CUH
void warp_affine_bilinear(
        /*
        建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
            - https://v.douyin.com/Nhre7fV/
         */
        uint8_t* src, int src_line_size, int src_width, int src_height,
        float* dst, int dst_line_size, int dst_width, int dst_height,
        uint8_t fill_value
);
#endif //FCOS_AFFINE_CUH
