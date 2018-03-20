#if 1
#include <iostream>
#include <opencv2/opencv.hpp>
#include "helper_timer.h"
#include "tr_detect.h"
#include "svm.h"
#include "svm-predict.h"
using namespace cv;
using namespace std;

char outputVote(char type[]);

int call(const Mat& imgl, const Mat& imgr, int outLabel,Mat& sResult, TRDetect& trd,int label=2)
{
	

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	//声明样本类别，初始化label
	//asphalt(120,120,120),grass(0,240,240),sand(240,240,0);
	
	char terrainType;
	//Mat sResult;

	//创建一个数组，存的是前面几帧的label
	char type[10]={'0'};
			
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    //第一次处理图像
    trd.process(imgl,imgr);
    //trd.addLabel(label);
    //保存标注以后的可行域图片
    //imwrite(path +"new/" +sa +"labeled"+ ".jpg",trd.addLabel(label));

    //显示结果
    sResult=trd.getSresult();
    //保存叠加结果
    //imwrite(path +"new/" + sa +"result"+ ".jpg",sResult);
    //用extract提取测试集的特征,并保存到feature.txt
    //trd.extract(label);

    //第二次处理
    trd.simpleExtract2(imgl,label);
    //使用svmpredict来对测试集预测，后续还要在线显示
    //argv命令参数,参数以字符串数组的形式传入
    char *argv[] = {"", "feature.txt", "train0131_true.model", "output.txt"};  //groundtrue train
    //char *argv[] = {"", "feature.txt", "train0301_200.model", "output.txt"};
    svmPredict(4,argv);
		
    //根据output的结果输出可行域类别分类结果
    //写在outputVote.cpp里面
    terrainType=outputVote(type);
    //将输出的分类结果显示在图片结果上
    switch (terrainType)
    {
        case 'a':
            putText(sResult,"asphalt",Point(20, int(sResult.rows*0.9)), FONT_HERSHEY_PLAIN,sResult.cols/160, cvScalar(0, 0, 200, 0));
			outLabel=1;
			break;
		case 'g':
            putText(sResult,"grass",Point(20, int(sResult.rows*0.9)), FONT_HERSHEY_PLAIN,sResult.cols/160, cvScalar(0, 0, 200, 0));
			outLabel=2;
			break;
		case 's':
			putText(sResult,"sand",Point(20, int(sResult.rows*0.9)), FONT_HERSHEY_PLAIN,sResult.cols/160, cvScalar(0, 0, 200, 0));
			outLabel=3;
			break;
		default:
			putText(sResult,"unknown",Point(20, int(sResult.rows*0.9)), FONT_HERSHEY_PLAIN,sResult.cols/160, cvScalar(0, 0, 200, 0));
			outLabel=0;
			break;
    }

    namedWindow("result",CV_WINDOW_NORMAL);
    imshow("result",sResult);
    waitKey(1);
    sdkStopTimer(&timer);
    printf("time spent: %.2fms\n", sdkGetTimerValue(&timer));
		
    return outLabel;
}

int main()
{

	//更改数据集的地址
    //string path = "/home/zc/Downloads/datesets/data2018/left/";
    //string path2= "/home/zc/Downloads/datesets/data2018/right/";
    string path= "/home/zc/Downloads/datesets/bumblebee2image/left/";
    string path2= "/home/zc/Downloads/datesets/bumblebee2image/right/";
//    string path = "/home/zc/Downloads/datesets/data2018/g/";
//    string path2= "/home/zc/Downloads/datesets/data2018/g/";
	TRDetect::parameters param;

	//设置一系列参数，这部分不能参与循环
	param.method = TRDetect::TRD_METHOD_STEREO;
	param.imgScale = 0.325f;
	//1.读取图片及其尺寸
    //param.imgSize = imgl.size();
	//2.或者在知道尺寸的情况下直接size=（640，480）
    param.imgSize = cv::Size(1024,768);

    param.calib.cu = 160.718;
    param.calib.cv = 122.632;
    //相机光轴在图像坐标系中的偏移量,以像素为单位
    param.calib.f  = 245.83;
	param.calib.baseline = 0.120014;

	int numScale = 3;
	int nSuperpixels[] = {450,180,65};
    //这是超像素的什么参数?
	float wVote[] = {0.7f,1.0f,0.7f};
    //这是什么投票比例?

	vector<int> k(nSuperpixels, nSuperpixels + sizeof(nSuperpixels) / sizeof(int));
	vector<float> w(wVote, wVote + sizeof(wVote) / sizeof(float));
	param.seg.numScale = numScale;
    //要分割的超像素总数
	param.seg.k = k;
	param.seg.weight = w;

	param.net.elm_af = ELM::ELM_AF_SIGMOID;

    TRDetect trd(param);//init TRDetect实例化

	

	Mat sResult;
	int sout;

    //创建一个数组，存的是前面几帧的label
	char type[10]={'0'};
    //char型指针,type是指针的基类型,他必须是一个有效的c++ 的数据类型.
    //所有指针的值的实际数据类型都是一个代表内存地址的长的十六进制数.

    for(int i=1; i<8074; i=i+7) //step into the loopstring path2= "/home/zc/Downloads/datesets/bumblebee2image/right"
	{
		//to fix the problem the filename
		//sa是序号，filename是特征提取后保存的文件名
		char a[10];
		string sa;
        sprintf(a,"%04d", i); //字符串格式化,把格式化的数据写入某个字符串中.
		sa =a;
        //std::cout << sa << std::endl;
		
        Mat imgl = imread(path + sa + ".jpg");
        Mat imgr = imread(path2 + sa + ".jpg");

        //直方图均衡
//        Mat imageRGB[3];
//        split(imgl, imageRGB);
//        for (int i = 0; i < 3; i++)
//        {
//          equalizeHist(imageRGB[i], imageRGB[i]);
//        }
//        merge(imageRGB, 3, imgl);
//        Mat image1RGB[3];
//        split(imgr, image1RGB);
//        for (int i = 0; i < 3; i++)
//        {
//          equalizeHist(image1RGB[i], image1RGB[i]);
//        }
//        merge(image1RGB, 3, imgr);

        call(imgl, imgr, sout, sResult,trd,1);
	}
}
#endif
