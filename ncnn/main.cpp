#include "net.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <map>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define INPUT_WIDTH     224
#define INPUT_HEIGHT    224

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("illegal parameters!");
        exit(0);
    }

    ncnn::Net Unet;

    Unet.load_param("/home/pc/contest/Keras-Semantic-Segmentation-master/ncnn/models/unet.param");
    Unet.load_model("/home/pc/contest/Keras-Semantic-Segmentation-master/ncnn/models/unet.bin");

    int64 tic, toc;

    tic = cv::getTickCount();

    cv::Scalar value = Scalar(0,0,0);
    cv::Mat src;
    cv::Mat tmp;
    src = cv::imread(argv[1]);
    float width = src.size().width;
    float height = src.size().height;
    int top = 0, bottom = 0;
    int left = 0, right = 0;

    if (width > height) {
        top = (width - height) / 2;
        bottom = (width - height) - top;
        cv::copyMakeBorder(src, tmp, top, bottom, 0, 0, BORDER_CONSTANT, value);
    } else {
        left = (height - width) / 2;
        right = (height - width) - left;
        cv::copyMakeBorder(src, tmp, 0, 0, left, right, BORDER_CONSTANT, value);
    }

    top = (INPUT_HEIGHT*top)/width;
    bottom = (INPUT_HEIGHT*bottom)/width;
    left = (INPUT_WIDTH*left)/height;
    right = (INPUT_WIDTH*right)/height;

    std::cout << "top " << top << " bottom " << bottom << " left " << left << " right " << right << std::endl;

    cv::Mat tmp1;
    cv::resize(tmp, tmp1, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), CV_INTER_CUBIC);

    cv::Mat image;
    tmp1.convertTo(image, CV_32FC3);

    std::cout << "image element type "<< image.type() << " " << image.cols << " " << image.rows << std::endl;

    // cv32fc3 的布局是 hwc ncnn的Mat布局是 chw 需要调整排布
    float *srcdata = (float*)image.data;
    float *data = new float[INPUT_WIDTH*INPUT_HEIGHT*3];
    for (int i = 0; i < INPUT_HEIGHT; i++)
       for (int j = 0; j < INPUT_WIDTH; j++)
           for (int k = 0; k < 3; k++) {
              data[k*INPUT_HEIGHT*INPUT_WIDTH + i*INPUT_WIDTH + j] = srcdata[i*INPUT_WIDTH*3 + j*3 + k]/255.0;
           }
    ncnn::Mat in(image.rows*image.cols*3, data);
    in = in.reshape(image.rows, image.cols, 3);

    ncnn::Extractor ex = Unet.create_extractor();

    ex.set_light_mode(true);
    //ex.set_num_threads(4);

    ex.input("data", in);

    ncnn::Mat mask;
    ex.extract("reshape_1_activation_21", mask);

    {
        toc = cv::getTickCount() - tic;

        double time = toc / double(cv::getTickFrequency());
        double fps = double(1.0) / time;
        std::cout << "fps:" << fps << std::endl;
    }

    std::cout << "whc " << mask.w << " " << mask.h << " " << mask.c << std::endl;
#if 1
    cv::Mat cv_img = cv::Mat::zeros(INPUT_WIDTH,INPUT_HEIGHT,CV_8UC1);
//    mask.to_pixels(cv_img.data, ncnn::Mat::PIXEL_GRAY);

    {
    float *srcdata = (float*)mask.data;
    unsigned char *data = cv_img.data;

    for (int i = 0; i < mask.h; i++)
       for (int j = 0; j < mask.w; j++) {
#if 1
         float tmp = srcdata[0*mask.w*mask.h+i*mask.w+j];
         int maxk = 0;
         for (int k = 0; k < mask.c; k++) {
           if (tmp < srcdata[k*mask.w*mask.h+i*mask.w+j]) {
             tmp = srcdata[k*mask.w*mask.h+i*mask.w+j];
             maxk = k;
           }
           //std::cout << srcdata[k*mask.w*mask.h+i*mask.w+j] << " ";
         }
         //cout << endl;
         data[i*INPUT_WIDTH + j] = maxk;

         if ((left > 0) && (right > 0) && ((j < left) || (j >= INPUT_WIDTH - right)))
           data[i*INPUT_WIDTH + j] = 0;

         if ((top > 0) && (bottom > 0) && ((i < top) || (i >= INPUT_HEIGHT - bottom)))
           data[i*INPUT_WIDTH + j] = 0;
#else
         if (srcdata[1*mask.w*mask.h+i*mask.w+j] > 0.999)
           data[i*INPUT_WIDTH + j] = 1;
         else
           data[i*INPUT_WIDTH + j] = 0;
#endif
       }
    }

    {
        toc = cv::getTickCount() - tic;

        double time = toc / double(cv::getTickFrequency());
        double fps = double(1.0) / time;
        std::cout << "fps:" << fps << std::endl;
    }

    cv_img *= 255;
    cv::imshow("test", cv_img);
    cv::imwrite("./res.jpg", cv_img);
    cv::waitKey();
#endif
    return 0;
}