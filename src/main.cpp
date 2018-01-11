#include <iostream>
#include <opencv2/opencv.hpp>
#include "helper_timer.h"
#include "tr_detect.h"
#include "svm.h"
//#include "outputVote.cpp"

using namespace cv;
using namespace std;

char outputVote(char type[]);

int main()
{
	TRDetect::parameters param;

	param.method = TRDetect::TRD_METHOD_STEREO;
	param.imgScale = 0.325f;
	//每次要改变图像尺寸啦！！！
	param.imgSize = Size(752,480);

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

	//更改数据库的复制起始点
	string path = "F:/dataset/part_a/wet-a/camera_stereo_left/frames/";
	string path2= "F:/dataset/part_a/wet-a/camera_stereo_right/frames/";



	int step = 1;
	//the number for the txt file
	int line = 0;
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	//声明样本类别，初始化label
	//asphalt(120,120,120),grass(0,240,240),sand(240,240,0);
	int label=2;
	char terrainType;
	Mat sResult;

	//创建一个数组，存的是前面几帧的label
	char type[10]={'0'};

	for(int i=100; i<400; i=i+step)  //step into the loop
	{
		//to fix the problem the filename
		//sa是序号，filename是特征提取后保存的文件名
		char a[10];
		string sa;
		sprintf(a,"%04d", i);
		sa =a;

		char filename[100]={0};
		string file = sa+"feature"+".txt";
		const char* chfile = file.c_str();
		strcpy(filename,chfile);


		Mat imgl = imread(path + sa + ".jpg");
		Mat imgr = imread(path2 + sa + ".jpg");
	    //Mat imgl = imread(path + "ImgL"+ to_string(i) + ".png");
	    //Mat imgr = imread(path + "ImgR"+ to_string(i) + ".png");
		
		//更改数据库的复制终止点


		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
		trd.process(imgl,imgr);
		trd.addLabel(label);
		//保存标注以后的可行域图片
		//imwrite(path +"new/" +sa +"labeled"+ ".jpg",trd.addLabel(label));
		//显示结果
		sResult=trd.getSresult();
		//保存叠加结果
		//imwrite(path +"new/" + sa +"result"+ ".jpg",sResult);
		//用extract提取测试集的特征,并保存到txt
		trd.extract(i,label);

		//使用svmpredict来对测试集预测，后续还要在线显示
		//判断sa+feature文件是否存在，若不存在，跳过predict
		//直接使用之前的output文件
		FILE *fpFeature=NULL;//需要注意
		fpFeature=fopen(filename,"r"); 
		if(NULL==fpFeature) continue;
		else{
			char *argv[] = {"", filename, "train0319.model", "output.txt"};
			svmPredict(4,argv);
		}

		//根据output的结果输出可行域类别分类结果
		//写在outputVote.cpp里面
		terrainType=outputVote(type);
		switch (terrainType)
		{
		case 'a':
			putText(sResult,"asphalt",Point(20, int(sResult.rows*0.9)), FONT_HERSHEY_PLAIN,sResult.cols/160, cvScalar(0, 0, 200, 0));
			break;
		case 'g':
			putText(sResult,"grass",Point(20, int(sResult.rows*0.9)), FONT_HERSHEY_PLAIN,sResult.cols/160, cvScalar(0, 0, 200, 0));
			break;
		case 's':
			putText(sResult,"sand",Point(20, int(sResult.rows*0.9)), FONT_HERSHEY_PLAIN,sResult.cols/160, cvScalar(0, 0, 200, 0));
			break;
		default:
			putText(sResult,"unknown",Point(20, int(sResult.rows*0.9)), FONT_HERSHEY_PLAIN,sResult.cols/160, cvScalar(0, 0, 200, 0));
			break;
		}
		//putText(sResult,"type",Point(20, int(sResult.rows*0.9)), FONT_HERSHEY_PLAIN,sResult.cols/200, cvScalar(0, 0, 200, 0));
		imshow("result",sResult);
		waitKey(1);
		sdkStopTimer(&timer);
		printf("time spent: %.2fms\n", sdkGetTimerValue(&timer));
		

		//save the results as jpg format(now it's useless)
		///////////////////////////////
		//Mat result;
		//trd.getTReigon(result);
		//cv::imwrite("result"+to_string(i)+".jpg",result);
		
		//extract training data by SimpleExtract
		//////////////////
		/*Mat timg = imread(path2+ "g" + to_string(i) + ".jpg");
		trd.simpleExtract(timg,i);
		*/

	}
	waitKey(0);
}