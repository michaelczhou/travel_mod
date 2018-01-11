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
		trd.process(imgl,imgr);
		//trd.addLabel(label);
		//保存标注以后的可行域图片
		//imwrite(path +"new/" +sa +"labeled"+ ".jpg",trd.addLabel(label));
		//显示结果
		sResult=trd.getSresult();
		//保存叠加结果
		//imwrite(path +"new/" + sa +"result"+ ".jpg",sResult);
		//用extract提取测试集的特征,并保存到feature.txt
		trd.extract(label);

		//使用svmpredict来对测试集预测，后续还要在线显示
		char *argv[] = {"", "feature.txt", "train0319.model", "output.txt"};
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
		//putText(sResult,"type",Point(20, int(sResult.rows*0.9)), FONT_HERSHEY_PLAIN,sResult.cols/200, cvScalar(0, 0, 200, 0));
		imshow("result",sResult);
		waitKey(1);
		sdkStopTimer(&timer);
		printf("time spent: %.2fms\n", sdkGetTimerValue(&timer));
		
		return outLabel;
}
int main(){

	//更改数据集的地址
    string path = "/home/zc/Downloads/datesets/Kitti/sequences/00/image_0/";
    string path2= "/home/zc/Downloads/datesets/Kitti/sequences/00/image_1/";

	TRDetect::parameters param;

	//设置一系列参数，这部分不能参与循环
	param.method = TRDetect::TRD_METHOD_STEREO;
	param.imgScale = 0.325f;
	//1.读取图片及其尺寸
	//param.imgSize = imgl.size();
	//2.或者在知道尺寸的情况下直接size=（640，480）
    param.imgSize = cv::Size(1241,376);

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

	TRDetect trd(param);

	

	Mat sResult;
	int sout;

    //创建一个数组，存的是前面几帧的label
	char type[10]={'0'};

    for(int i=1000; i<4000; i++) //step into the loop
	{
		//to fix the problem the filename
		//sa是序号，filename是特征提取后保存的文件名
		char a[10];
		string sa;
        sprintf(a,"%06d", i);
		sa =a;
        //std::cout << sa << endl;
		
        Mat imgl = imread(path + sa + ".png");
        Mat imgr = imread(path2 + sa + ".png");

        call(imgl, imgr, sout, sResult,trd,1);
	}
}
