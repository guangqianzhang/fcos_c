#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "preprocess.h"
using namespace cv;
using namespace std;
Mat image_read(const std::string& file_name, int flag=IMREAD_COLOR){
    Mat image=cv::imread(file_name,flag=flag);
    return image;
}
void printFloatArrayAddress(float* float_array,int size) {

    std::cout << "Contents of the float array:" << std::endl;
    for (int i = 0; i < size; ++i) {
        std::cout << float_array[i] << "\t";
    }
    std::cout << std::endl;
}
int main() {
    std::cout << "Hello, World!" << std::endl;
    std::string file_name="/home/cqjtu/Pictures/CAM_FRONT__1526915630862465.jpg";
    Mat back_image= image_read(file_name);
    cv::Size size(640, 640);
    float *output = warpaffine_to_center_align(back_image, size);
//    printFloatArrayAddress(output,size.width*size.height);
    // 创建一个cv::Mat变量，指向float_container的数据
    cv::Mat mat(size , CV_32FC3, output);

    // 释放动态分配的内存
    delete[] output;
//    cv::imshow("show",output);
//    cv::waitKey(1000);
    std::cout<<mat<<endl;
    cv::imwrite("normalize.jpg", mat);
    return 0;
}
