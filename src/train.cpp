#if 0
#include <iostream>
#include <opencv2/opencv.hpp>
#include "helper_timer.h"
#include "tr_detect.h"
#include "svm.h"
#include "svm-predict.h"

using namespace cv;
using namespace std;

int main()
{
    TRDetect::parameters param;

    param.method = TRDetect::TRD_METHOD_MONO;
    param.imgScale = 0.325f;
    //每次要改变图像尺寸啦！！！
    param.imgSize = Size(208,156);
    //param.imgSize = Size(1024,768);  //车采集
    //param.imgSize = imgl.size();

    param.calib.cu = 321.93585;
    param.calib.cv = 245.76448;
    param.calib.f  = 491.659520;
    param.calib.baseline = 0.120014;

    int numScale = 3;
    int nSuperpixels[] = {450,180,65};
    float wVote[] = {0.7f,1.0f,0.7f};

    vector<int> k(nSuperpixels, nSuperpixels + sizeof(nSuperpixels) / sizeof(int));
    vector<float> w(wVote, wVote + sizeof(wVote) / sizeof(float));
    param.seg.numScale = numScale;
    param.seg.k = k;
    param.seg.weight = w;

    param.net.elm_af = ELM::ELM_AF_SIGMOID;
    //init TRDetect
    TRDetect trd(param);

    string path = "/home/zc/Downloads/datesets/data2018/g/";

    int num = 1;

    for(int i = 1;i < 441;i++)
    {
        char a[10];
        string sa;
        sprintf(a,"%06d",i);
        sa = a;

        Mat img = imread(path + sa + ".jpg");

//        Mat imageRGB[3];
//        split(img, imageRGB);
//        for (int i = 0; i < 3; i++)
//        {
//          equalizeHist(imageRGB[i], imageRGB[i]);
//        }
//        imerge(imageRGB, 3, img);
        //imshow("show",imageROI);
        trd.simpleExtract(img,num);
        num++;

        waitKey(1);


    }
    return 0;
}

#endif
