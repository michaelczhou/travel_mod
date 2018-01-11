#ifndef TR_DECTECT_H
#define TR_DECTECT_H

#include "opencv2/core/core.hpp"
#include "elm.h"


class TRDetect
{
private:
    struct _camera // carema parameters, for stereo method
    {
        double f;
        double cu;
        double cv;
        double baseline;
        int SADWindowSize;
        int minDisparity;
        int numberOfDisparities;
        _camera()
        {
            f = 0.0;
            cv = 0.0;
            cv = 0.0;
            baseline = 0.0;
            SADWindowSize = 11;
            minDisparity = 3;
            numberOfDisparities = 48;
        }
    };
    struct _fm // fundamentalMask config
    {
        float able_h,able_w;
        float unable_h,unable_w;
        _fm()
        {
            able_h = 0.25f;
            able_w = 0.30f;
            unable_h = 0.20f;
            unable_w = 0.20f;
        }
    };
    struct _seg // segment parameters
    {
        int numItr;                // number of iteration when performing SLIC
        int numScale;              // number of scale, should be odd
        std::vector<int> k;        // number of superpixels for each scale
        std::vector<float> weight; // weight for each scale when training
        _seg()
        {
            numScale = 3;
            numItr = 5;
        }
    };
    class network // neural network parameters
    {
    //TODO: nBin_H,nBin_S,nBin_V can be changed
    private:
        const int nBin_LBP;
        const int nBin_H;
        const int nBin_S;
        const int nBin_V;
        const int nInputNeurons;
        const int nOutputNeurons;
        friend TRDetect;
    public:
        int nHiddenNeurons;
        int elm_af;
        float lambda;
        float sigma;
        float roadPriority;
        network():
        nBin_LBP(10),
        nBin_H(18),
        nBin_S(18),
        nBin_V(9),
        nInputNeurons(nBin_LBP + nBin_H + nBin_S + nBin_V),
        nOutputNeurons(2)
        {
            elm_af = ELM::ELM_AF_SIGMOID_FAST;
            nHiddenNeurons = 55;
            lambda = 0.8f;
            sigma = 2.75f;
            roadPriority = 0.05f;
        }
    };
    struct layerFeature
    {
        int nSuperpixels;
        cv::Mat imgSeg;    //      uint
        cv::Mat imgProb;   //      float
        cv::Mat info;      // N*3  uint 
        cv::Mat input;     // N*55 float
        cv::Mat output;    // N*2  float
        cv::Mat weight;    // N*1  float
		cv::Mat flagTR;    // N*1  unit
        layerFeature()
        {
            nSuperpixels = 0;
        }
    };

    enum
    {
        INFO_SZ,
        INFO_ROW,
        INFO_COL,
        INFO_NUM,
    };

public:
    enum
    {
        TRD_METHOD_MONO,
        TRD_METHOD_STEREO,
        TRD_METHOD_KINECT
    };

    struct parameters
    {
        int method;

        cv::Size imgSize;
        float imgScale;

        _fm fm;
        _camera calib;
        _seg seg;
        network net;
        parameters()
        {
            method = TRD_METHOD_STEREO;
            imgSize = cv::Size(640,480);
            imgScale = 0.325f;
        }
    };

public:
    TRDetect(parameters param);
    ~TRDetect();

    // the second parameter can be Mat(), imgR or pointCloud
    bool process(const cv::Mat &imageL, const cv::Mat &imageR);
	void extract(int label);  //get testing data
	void simpleExtract(const cv::Mat &img, int num);  //get training data

    void stereoMatch(const cv::Mat &imageL, const cv::Mat &imageR, cv::Mat& disparity, cv::Mat& pointCloud);

    void getImageMono(cv::Mat &imageMono);
    void getImageColor(cv::Mat &imageColor);
    void getTReigon(cv::Mat &TReigon);
    void getTReigonProb(cv::Mat &TReigonProb);
    void getGroundTruth(cv::Mat &groundTruth);
    cv::Mat getSresult();
	cv::Mat addLabel(int label);

private:
    parameters param;
    cv::Size sz;

    cv::Mat imgColorL;       // available for all method
    cv::Mat imgColorR;
    cv::Mat imgMonoL;        // available for all method
    cv::Mat imgMonoR;
    cv::Mat disparity;       // only available for TRD_METHOD_STEREO
    cv::Mat pointCloud;      // not available for TRD_METHOD_MONO

    cv::Mat groundTruth;     // CV_8U
    cv::Mat fundamentalMask; // CV_8U
    cv::Mat result;          // CV_8U
	
    cv::Mat resultProb;      // CV_32F

    ELM* elm;
    cv::Mat outputWeight;
    cv::Mat imgHSV;
    cv::Mat imgLBP;
    layerFeature* layer;
    layerFeature  layersCur;
    layerFeature  layersPool;

private:
    bool updateGroundTruth();
    void updateLayer();
    void updatelayersCur();
    void updateLayersPool();
    void rectifyWeight();
    void rebuildTRegion();
    void doSegmentation();
	
};

#endif
