//Traversable Region Detection
#include "tr_detect.h"
#include "iostream"
#include "cassert"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "Eigen/Dense"
#include "ransac_plain.h"
#include "lbp.h"
#include "slic.h"
#include "cvblob2.h"
#ifdef __CUDACC__
#include "gslic/FastImgSeg.h"
#endif

using namespace cv;
using namespace std;
using namespace Eigen;

#define TY 0xFF // traversable
#define TN 0x7F // impassable
#define TU 0x00 // unknown

TRDetect::TRDetect(parameters param):
param(param)
{
    // check carema params have being set
    if(param.method == TRD_METHOD_STEREO)
    {
        assert((param.calib.f != 0.0) & 
               (param.calib.cu != 0.0) &
               (param.calib.cv != 0.0) &
               (param.calib.baseline != 0.0));
    }

    // check nScale 
    const int nScale = param.seg.numScale;
    //assert(nScale&0x01); // must be odd

    // check k and weight have being set
    assert((param.seg.k.size() == nScale ) & (param.seg.weight.size() == nScale));

    // initialize some pointer
    elm = new ELM(55,50,ELM::ELM_AF_SIGMOID_FAST);
    layer = new layerFeature[nScale];

    // initialize size
    sz.height = (int)(param.imgSize.height * param.imgScale + 0.5f);
    sz.width  = (int)(param.imgSize.width  * param.imgScale + 0.5f);

    // initialize the fundamental Mask
    fundamentalMask = Mat::zeros(sz,CV_8U);
    int w_able = (int)(param.fm.able_w * sz.width + 0.5f);      // traversable region width
    int h_able = (int)(param.fm.able_h * sz.height + 0.5f);     // traversable region height
    int w_unable = (int)(param.fm.unable_w * sz.width + 0.5f);  // untraversable region width
    int h_unable = (int)(param.fm.unable_h * sz.height + 0.5f); // untraversable region height
    Mat ROI1(fundamentalMask,Rect((sz.width-w_able)/2, sz.height-h_able, w_able, h_able));
    ROI1=Scalar(TY);                // roi1 force to traversable
    Mat ROI2(fundamentalMask,Rect(0,                 0, w_unable, h_unable));
    ROI2=Scalar(TN);                // Roi2 force to untraversable
    Mat ROI3(fundamentalMask,Rect(sz.width-w_unable, 0, w_unable, h_unable));
    ROI3=Scalar(TN);                // Roi3 force to untraversable
}

TRDetect::~TRDetect()
{
    delete elm;
    delete []layer;
}

bool TRDetect::process(const Mat &image)
{
    // Get image ready
    // --------------------------------------------------

    // imgColorL
    // pointCloud
    int rsMethod = param.imgScale < 1.0f ? INTER_AREA:INTER_CUBIC;
    resize(image, imgColor, Size(), param.imgScale, param.imgScale, rsMethod);
    cvtColor(imgColor, imgMono, CV_BGR2GRAY);

    if(param.method != TRD_METHOD_MONO)
    pointCloud = image.clone();


    // get ground truth for each method
    // TRD_METHOD_MONO   - fundamentalMask
    // TRD_METHOD_STEREO - imgMonoL, imgMonoR
    // TRD_METHOD_KINECT - pointCloud
    bool flagTrain = updateGroundTruth();

    // Mat addd;
    // RANSAC::cvtMaskn3(groundTruth, addd);
    // Mat show = 0.75*imgColorL+ 0.25*addd;
    // imshow("show",show);

    /////////////////////////////////////////////////////////////////////////
    // Now we get imgColorL, imgMonoL, groundTruth and model for all method
    // Start processing ...

    // Get features
    cvtColor(imgColor, imgHSV, CV_BGR2HSV); // HSV
    lbpRiu2(imgMono, imgLBP);               // LBP
    doSegmentation();                        // SLIC

    // Extract data from feature
    updateLayer();

    // ELM train if outputWeight is empty
    if(outputWeight.empty())
    {
        if(!flagTrain)
            return false;

        // Update layersCur and rectify weight
        updatelayersCur();
        rectifyWeight();

        // ELM train
        elm->train(layersCur.input, layersCur.output, outputWeight, layersCur.weight);
    }

    // ELM predict
    for(int n=0; n<param.seg.numScale; ++n)
    {
        elm->predict(layer[n].input, layer[n].output, outputWeight);
    }

    // Rebuild traversable region
    rebuildTRegion();

    // Mat sss = 0.75*imgMonoL+0.25*result;
    // imshow("res",result);

    ///////////////////////
    ////////////////////
    ////////////
    //add
    ////

    if(!flagTrain)
        return false;

    // Update layersCur and rectify weight
    updatelayersCur();
    rectifyWeight();

    // Update layersPool
    updateLayersPool();

    // ELM train
    // cout<<layersPool.input.rows<<endl;
    elm->train(layersPool.input, layersPool.output, outputWeight, layersPool.weight);

    return true;
}


bool TRDetect::process(const Mat &imageL,const Mat &imageR)
{
    // Get image ready
    // --------------------------------------------------
    if(param.method == TRD_METHOD_STEREO)
    {
        // imgColorL imgColorR
        // imgMonoL  imgMonoR
        int rsMethod = param.imgScale < 1.0f ? INTER_AREA:INTER_CUBIC;
        resize(imageL, imgColorL, Size(), param.imgScale, param.imgScale, rsMethod);
        resize(imageR, imgColorR, Size(), param.imgScale, param.imgScale, rsMethod);
        cvtColor(imgColorL, imgMonoL, CV_BGR2GRAY);
        cvtColor(imgColorR, imgMonoR, CV_BGR2GRAY);
    }
    else
    {
        // imgColorL
        // pointCloud
        int rsMethod = param.imgScale < 1.0f ? INTER_AREA:INTER_CUBIC;
        resize(imageL, imgColorL, Size(), param.imgScale, param.imgScale, rsMethod);
        cvtColor(imgColorL, imgMonoL, CV_BGR2GRAY);

        if(param.method != TRD_METHOD_MONO)
            pointCloud = imageR.clone();
    }

    // get ground truth for each method
    // TRD_METHOD_MONO   - fundamentalMask
    // TRD_METHOD_STEREO - imgMonoL, imgMonoR
    // TRD_METHOD_KINECT - pointCloud
    bool flagTrain = updateGroundTruth();

    // Mat addd;
    // RANSAC::cvtMaskn3(groundTruth, addd);
    // Mat show = 0.75*imgColorL+ 0.25*addd;
    // imshow("show",show);

    /////////////////////////////////////////////////////////////////////////
    // Now we get imgColorL, imgMonoL, groundTruth and model for all method
    // Start processing ...

    // Get features
    cvtColor(imgColorL, imgHSV, CV_BGR2HSV); // HSV
    lbpRiu2(imgMonoL, imgLBP);               // LBP
    doSegmentation();                        // SLIC

    // Extract data from feature
    updateLayer();

    // ELM train if outputWeight is empty
    if(outputWeight.empty())
    {
        if(!flagTrain)
            return false;
        
        // Update layersCur and rectify weight
        updatelayersCur();
        rectifyWeight();

        // ELM train
        elm->train(layersCur.input, layersCur.output, outputWeight, layersCur.weight);
    }

    // ELM predict
    for(int n=0; n<param.seg.numScale; ++n)
    {
        elm->predict(layer[n].input, layer[n].output, outputWeight);
    }

    // Rebuild traversable region
    rebuildTRegion();
	
    // Mat sss = 0.75*imgMonoL+0.25*result;
    // imshow("res",result);

	///////////////////////
	////////////////////
	////////////
	//add
	////

    if(!flagTrain)
        return false;

    // Update layersCur and rectify weight
    updatelayersCur();
    rectifyWeight();

    // Update layersPool
    updateLayersPool();

    // ELM train
    // cout<<layersPool.input.rows<<endl;
    elm->train(layersPool.input, layersPool.output, outputWeight, layersPool.weight);

    return true;
}

void TRDetect::stereoMatch(const Mat &imageL,const Mat &imageR, Mat& disparity, Mat& pointCloud)
{
    StereoBM bm;
    float scale = param.imgScale;
    int SAD = param.calib.SADWindowSize; //int(param.calib.SADWindowSize * scale * 2 + 0.5f) | 0x01; // odd
    int minDisparity = (int)(param.calib.minDisparity * scale + 0.5f);
    int numberOfDisparities = (int(param.calib.numberOfDisparities * scale + 15.5f) & -16); // n*16
    bm.state->preFilterCap = 31;
    bm.state->SADWindowSize = SAD; //
    bm.state->minDisparity = minDisparity; //
    bm.state->numberOfDisparities = numberOfDisparities; //
    bm.state->textureThreshold = 10;
    bm.state->uniquenessRatio = 15;
    bm.state->speckleWindowSize = 100;
    bm.state->speckleRange = 32;
    bm.state->disp12MaxDiff = 1;

    // stereo match
    bm(imageL, imageR, disparity, CV_32F);
    // Mat disp;
    // disparity.convertTo(disp, CV_8U, 255/(numberOfDisparities+minDisparity));
    // imshow("disparity",disp);

    // reproject image to 3D
    // [x' y' z' W]' = Q*[x y disparity 1]'
    // X = x'/W
    // Y = y'/W
    // Z = z'/W
    // [1 0  0  -cu]
    // [0 1  0  -cv]
    // [0 0  0   f ]
    // [0 0 1/b  0 ]
    Mat Q = (Mat_<double>(4,4)<< 1.0, 0.0, 0.0, -param.calib.cu * scale,
                                 0.0, 1.0, 0.0, -param.calib.cv * scale,
                                 0.0, 0.0, 0.0,  param.calib.f * scale,
                                 0.0, 0.0, 1.0/param.calib.baseline, 0.0);
    reprojectImageTo3D(disparity, pointCloud, Q, true);
}

/**
 * get ground truth
 * @return a flag indicate whether the network should be trained
 */
bool TRDetect::updateGroundTruth()
{
    Vec4d model;
    Vec4d modelUndef;

    double DIST_THRESHOLD = 0.05;
    int SAMPLE_STRIDE = 4;

    if(param.method == TRD_METHOD_MONO)
    {
        groundTruth = fundamentalMask;
        return true;
    }

    if(param.method == TRD_METHOD_STEREO)
    {
        stereoMatch(imgMonoL, imgMonoR, disparity, pointCloud);
        SAMPLE_STRIDE = (sz.height+50-1)/50;
    }

    // Compute groundTruth from pointCloud
    Mat mask;
    model = RANSAC::getMonoMask(pointCloud, mask, sz, DIST_THRESHOLD, SAMPLE_STRIDE);

    //// model: ax+by+cz+d=0
    //// are b and d are valid, if not, DONNOT train
    //if(model[1]<0.85 || abs(model[3]+1.17)>0.1) // 1.07~1.27m
    //{
    //    groundTruth = fundamentalMask;
    //    return false;
    //}

    //// get the average model
    //static Vec4d avgModel;
    //if(avgModel[0] == 0) avgModel = model;
    //avgModel = 0.95*avgModel + 0.05*model;
    //double deltaA = avgModel[0]*model[0] + avgModel[1]*model[1] + avgModel[2]*model[2]; // delta theta
    //double deltaH = abs(model[3] - avgModel[3]);                                        // delta d

    //// cos(10)=0.985
    //if(deltaA < 0.985 || deltaH > 0.05)
    //{
    //    groundTruth = fundamentalMask;
    //    return false;
    //}

    int w_able = (int)(param.fm.able_w * sz.width + 0.5f);      // traversable region width
    int h_able = (int)(param.fm.able_h * sz.height + 0.5f);     // traversable region height
    int w_unable = (int)(param.fm.unable_w * sz.width + 0.5f);  // untraversable region width
    int h_unable = (int)(param.fm.unable_h * sz.height + 0.5f); // untraversable region height
    Mat ROI1(mask,Rect((sz.width-w_able)/2, sz.height-h_able, w_able, h_able));
    ROI1=Scalar(TY);                // roi1 force to traversable
    Mat ROI2(mask,Rect(0,                 0, w_unable, h_unable));
    ROI2=Scalar(TN);                // Roi2 force to untraversable
    Mat ROI3(mask,Rect(sz.width-w_unable, 0, w_unable, h_unable));
    ROI3=Scalar(TN);                // Roi3 force to untraversable

    // label the traversable region
    cvb::CvBlobs blobs;
    Mat imgLabel, imgTR, imgTR_filter;
    threshold(mask, imgTR, TN+1, TY, THRESH_BINARY);
    cvb::Label(imgTR, imgLabel, blobs);

    int nLabel = blobs.size();
 
    // if number of label is small, we accept the mask
    if(nLabel <= 2)
    {
        groundTruth = mask;
        return true;
    }

    // else get the label in the traversable region and extract the main traversable region
    int label = cvb::GetLabel(imgLabel, sz.width/2, sz.height-h_able/2);
    cvFilterByLabel(blobs, label);
    cvb::FilterLabels(imgLabel, imgTR_filter, blobs);

    if(blobs.at(label)->area < sz.width*sz.height/9)
    {
        groundTruth = imgTR_filter;
        return false;
    }

    Mat kernal = getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(10,10));
    morphologyEx(imgTR_filter, imgTR_filter, MORPH_OPEN, kernal);

    Mat ROI4(imgTR_filter,Rect(0,                 0, w_unable, h_unable));
    ROI4=Scalar(TN);                // Roi2 force to untraversable
    Mat ROI5(imgTR_filter,Rect(sz.width-w_unable, 0, w_unable, h_unable));
    ROI5=Scalar(TN);                // Roi3 force to untraversable

    // we also train the net work
    groundTruth = imgTR_filter;
    return true;
}

/**
 * do Segmentation
 * it has cpu and gpu implementation
 */
void TRDetect::doSegmentation()
{
    Mat tmp;
    //dataformat transformat
    cvtColor(imgColorL, tmp, CV_BGR2BGRA);
    const unsigned int* buff = (const unsigned int*)tmp.data;

    for(int n=0; n<param.seg.numScale; ++n)
    {
        int& nSuperpixels = layer[n].nSuperpixels;

        #ifndef __CUDACC__
        ////////////////////
        // cpu slic
        SLIC mySeg;
        layer[n].imgSeg.create(sz, CV_32S);
        Mat imgSeg = layer[n].imgSeg;

        mySeg.PerformSLICO_ForGivenK( buff, sz.width, sz.height, 
                                      (int*)imgSeg.data,
                                      nSuperpixels,
                                      param.seg.k[n],
                                      param.seg.numItr);
        #else
        ////////////////////
        // gpu slic
        float weight = 10.0f;
        FastImgSeg* mySeg = new FastImgSeg();
        mySeg->initializeFastSeg(sz.width, sz.height, param.seg.k[n]);
        mySeg->LoadImg((unsigned char*)buff);
        mySeg->DoSegmentation(LAB_SLIC, weight); // TODO: set parameters
        for(int i=0; i<sz.width*sz.height; ++i)
        {
            if( nSuperpixels < mySeg->segMask[i])
                nSuperpixels = mySeg->segMask[i];
        }
        nSuperpixels++;
        layer[n].imgSeg = cv::Mat(sz, CV_32S, mySeg->segMask, image.cols*sizeof(int)).clone();
        mySeg->~FastImgSeg();
        #endif

        //TODO:
        //label.convertTo(label,CV_16U);  //to morphology the output label,which do not surpport the CV_32S
        //morphologyEx(label,label,CV_MOP_OPEN,kernal);
        //label.convertTo(label,CV_32S);

        //seg.create(dsize,CV_8UC3); //to save image after segmentation
        //level.DrawContoursAroundSegmentsTwoColors(now_test,label.ptr<int>(), image.cols,image.rows);//for black-and-white contours around superpixels
        //int2mat(now_test,image.rows,image.cols,seg);
    }
}

/**
 * update layer[n]'s feature
 * layer[n].info   - Superpixels' size and centre
 * layer[n].input  - feature vector
 * layer[n].weight - initial weight set by user
 */
void TRDetect::updateLayer()
{
    const int nScale = param.seg.numScale;
    for(int n=0; n<nScale; ++n)
    {
        const int nSuperpixels = layer[n].nSuperpixels;
        assert(nSuperpixels!=0);

        layer[n].info.create(nSuperpixels, INFO_NUM, CV_32S);
        layer[n].input.create(nSuperpixels, param.net.nInputNeurons, CV_32F);
        layer[n].weight.create(nSuperpixels, 1, CV_32F);

        // create references and initialize
        Mat& imgSeg = layer[n].imgSeg;
        Mat& info   = layer[n].info;
        Mat& input  = layer[n].input;
        Mat& weight = layer[n].weight;
        info.setTo(Scalar(0));
        input.setTo(Scalar(0.0f));

        const int nBin_H = param.net.nBin_H;
        const int nBin_S = param.net.nBin_S;
        const int nBin_V = param.net.nBin_V;
        const int step1 = nBin_H;
        const int step2 = step1 + nBin_S;
        const int step3 = step2 + nBin_V;
        float scaleH = 180.0f/nBin_H;
        float scaleS = 256.0f/nBin_S;
        float scaleV = 256.0f/nBin_V;

        //////////////////
        // get data ready
        for(int r=0; r<sz.height; ++r)
        {
            const auto pHSV = imgHSV.ptr<const Point3_<unsigned char>>(r);
            const auto pLBP = imgLBP.ptr<const unsigned char>(r);
            const auto pSeg = imgSeg.ptr<const unsigned int>(r);
            for(int c=0; c<sz.width; ++c)
            {
                auto label = pSeg[c];
                auto pInput = (float *)input.ptr(label);

                int posH = (int)(pHSV[c].x/scaleH);
                int posS = (int)(pHSV[c].y/scaleS) + step1;
                int posV = (int)(pHSV[c].z/scaleV) + step2;
                int posLBP = pLBP[c] + step3;

                ++pInput[posH];
                ++pInput[posS];
                ++pInput[posV];
                ++pInput[posLBP];

                auto pInfo = (unsigned int *)info.ptr(label);
                ++pInfo[INFO_SZ];
                pInfo[INFO_ROW] += r;
                pInfo[INFO_COL] += c;
            }
        }

        // 1.input: normalize the network input
        // TODO: how to normalize?
        // Map<Matrix<float, Dynamic, Dynamic, RowMajor>> inputE((float *)input.data, input.rows, input.cols);
        // inputE = inputE.rowwise().normalized();
        for(int r=0; r<input.rows; ++r)
        {
            input.row(r) /= info.at<uint32_t>(r,INFO_SZ);
        }

        // 2.weight: initialize weight
        weight.setTo(Scalar(param.seg.weight[n]));

        // 3.compute the center of Superpixels
        info.col(INFO_ROW) /= info.col(INFO_SZ);
        info.col(INFO_COL) /= info.col(INFO_SZ);
    }
}

/**
 * update layersCur
 * layersCur.input   - merge layer[n].input
 * layersCur.output  - from groundTruth, filter result and elm predict
 * layersCur.weight  - merge layer[n].weight
 */
void TRDetect::updatelayersCur()
{
    Mat filter;
    if(result.empty())
        filter = groundTruth;
    else   
        filter = result;

    const Mat& truth = groundTruth;
    
    const int nScale = param.seg.numScale;
    Mat* arrayInput = new Mat[nScale];
    Mat* arrayOutput = new Mat[nScale];
    Mat* arrayWeight = new Mat[nScale];

    for(int n=0; n<nScale; ++n)
    {
        arrayInput[n]  = layer[n].input;
        arrayWeight[n] = layer[n].weight; 

        const int nSuperpixels = layer[n].nSuperpixels;
        assert(nSuperpixels!=0);

        Mat& info = layer[n].info;
        Mat& predict = layer[n].output;

        arrayOutput[n].create(nSuperpixels, param.net.nOutputNeurons, CV_32F);
        Mat& output = arrayOutput[n];
        output.setTo(Scalar(-1.0f));

        // output: get output according to groundTruth, filter result and elm predict
        int nTP = 0; // true positive
        int nFP = 0; // false positive
        int nTN = 0; // true negative
        int nFN = 0; // false negative
        const auto pInfo = (const Vec<unsigned int, INFO_NUM>*)info.data;
        const auto pOutput = (Point2f*)output.data;
        const auto pPredict = (Point2f*)predict.data;

        int select = 0;
        for(int i=0; i<nSuperpixels; ++i)
        {
            const int curRow = pInfo[i][INFO_ROW];
            const int curCol = pInfo[i][INFO_COL];

            auto flagGth = truth.at<unsigned char>(curRow, curCol);
            auto flagRusult = filter.at<unsigned char>(curRow, curCol);

            if(flagGth == TY)
            {
                if(flagRusult == TY)
                    ++nTP;
                else
                    ++nFP;

                pOutput[i].x = 1.0f;
            }
            else if(flagGth == TN)
            {
                if(flagRusult != TY) 
                    ++nTN;
                else
                    ++nFN;
                
                // because the ground true isn't the real ground true
                // so ...
                select++;
                if(select%4==0)
                    pOutput[i].y = 1.0f;
                else
                    pOutput[i].y = 0.0f;
            }
            else
            {
                if(pPredict == nullptr)
                    continue;

                if((flagRusult == TY) && (pPredict[i].x > 0.0f))
                    pOutput[i].x = 1.0f;
                else if((flagRusult != TY) && (pPredict[i].x < 0.0f))
                    pOutput[i].y = 1.0f;
                else
                    pOutput[i].x = 0.0f;
            }
        }
    }

    vconcat(arrayInput,  nScale, layersCur.input);
    vconcat(arrayOutput, nScale, layersCur.output);
    vconcat(arrayWeight, nScale, layersCur.weight);

    {
        int index = 0;
        Mat& input = layersCur.input;
        Mat& output = layersCur.output;
        Mat& weight = layersCur.weight;
        const auto pOutput = (const Point2f*)output.data;
        while((index != output.rows) && (output.rows != 1))
        {   
            if( pOutput[index].x == 0.0f || pOutput[index].y == 0.0f)
            {
                input.row(weight.rows-1).copyTo(input.row(index));
                output.row(weight.rows-1).copyTo(output.row(index));
                weight.row(weight.rows-1).copyTo(weight.row(index));
                input.pop_back();
                output.pop_back();
                weight.pop_back();
                continue;
            }
            ++index;
        } 
    }

    delete []arrayInput;
    delete []arrayOutput;
    delete []arrayWeight;
}

/**
 * update LayersPool
 * 1.decreasing the weight of last frame(LayersPool)
 * 2.discard the feature if its weight is small
 * 3.merge layersCur into LayersPool
 */
void TRDetect::updateLayersPool()
{
    Mat& curInput  = layersCur.input;
    Mat& curOutput = layersCur.output;
    Mat& curWeight = layersCur.weight;

    if(layersPool.weight.empty())
    {
        layersPool.input  = curInput.clone();
        layersPool.output = curOutput.clone();
        layersPool.weight = curWeight.clone();
        return;
    }

    Mat& input = layersPool.input;
    Mat& output = layersPool.output;
    Mat& weight = layersPool.weight;


    // 1.decreasing the weight of last frame(LayersPool)
    weight -= 0.15f;     // TODO: deltaW

    // 2.discard the feature if its weight is small
    int index=0;
    const auto pWeight = (const float*)weight.data;
    while((index != weight.rows) && (weight.rows != 1))
    {   
        if( pWeight[index] < 0.01f)
        {
            // always copy the last row 
            input.row(weight.rows-1).copyTo(input.row(index));
            output.row(weight.rows-1).copyTo(output.row(index));
            weight.row(weight.rows-1).copyTo(weight.row(index));

            // pop back the last row
            // (weight.rows != 1) guarantee we can use pop_back safely
            input.pop_back();
            output.pop_back();
            weight.pop_back();
            continue;
        }
        ++index;
    }

    // 3.merge layersCur into LayersPool
    input.push_back(curInput);
    output.push_back(curOutput);
    weight.push_back(curWeight);
}

/**
 * rectify weight
 * rectify weight of this frame(layersCur)
 * according to a function of the number of traversable region and impassable region
 */
void TRDetect::rectifyWeight()
{
    Mat& curInput  = layersCur.input;
    Mat& curOutput = layersCur.output;
    Mat& curWeight = layersCur.weight;

    const auto pOutput = (const Point2f*)curOutput.data;
    const auto pWeight = (float*)curWeight.data;

    // count the number of traversable region and impassable region
    int nPos = 0;
    int nNeg = 0;
    for(int i=0; i<curOutput.rows; ++i)
    {
        if(pOutput[i].x > 0.0f)
            ++nPos;
    }
    nNeg = curOutput.rows - nPos;

    // compute non-linear balance coefficient
    const float lambda = param.net.lambda;
    const float sigma = param.net.sigma;
    const float roadPriority = param.net.roadPriority;
    const float Cd = (float)(nPos - nNeg)/curOutput.rows;
    const float Cb = lambda * CV_SIGN(Cd) * pow(abs(Cd), sigma);

    // rectify weight
    for(int i=0; i<curOutput.rows; ++i)
    {
        pWeight[i] -= CV_SIGN(pOutput[i].x) * (Cb - roadPriority);
    }
}

/**
 * rebuild TRegion
 * 
 */
void TRDetect::rebuildTRegion()
{
    const int nScale = param.seg.numScale;

    resultProb.create(sz, CV_32F);
    resultProb.setTo(Scalar(0.0f));

    for(int n=0; n<nScale; ++n)
    {
        layer[n].imgProb.create(sz, CV_32F);

        Mat& Seg = layer[n].imgSeg;
        Mat& output = layer[n].output;
        Mat& Prob = layer[n].imgProb;

        const auto pOutput = (const Point2f*)output.data;
        for(int r=0; r<sz.height; ++r)
        {
            const auto pSeg  = (const unsigned int*)Seg.ptr(r);
            const auto pProb = (float*)Prob.ptr(r);
            for(int c=0; c<sz.width; ++c)
            {
                auto label = pSeg[c];
                pProb[c] = pOutput[label].x;
            }
        }

        resultProb += Prob;
    }

    // Mat vote = Mat( sz, CV_32F, Scalar(-(nScale/2+1)*(TY-1)) );
    // for(int n=0; n<nScale; ++n)
    // {
    //     threshold(layer[n].imgProb, layer[n].imgProb, 0.01f, TY, THRESH_BINARY);
    //     vote += layer[n].imgProb;
    // }

    // threshold(vote, vote, TY, TY, THRESH_BINARY);
    // vote.assignTo(result, CV_8U);
	Mat iiii=resultProb;
    Mat vt;
    threshold(resultProb, vt, 0.02f, TY, THRESH_BINARY);
    vt.convertTo(result, CV_8U);

    int w_able = (int)(param.fm.able_w * sz.width + 0.5f);      // traversable region width
    int h_able = (int)(param.fm.able_h * sz.height + 0.5f);     // traversable region height
    int w_unable = (int)(param.fm.unable_w * sz.width + 0.5f);  // untraversable region width
    int h_unable = (int)(param.fm.unable_h * sz.height + 0.5f); // untraversable region height

    Mat kernal = getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(15,15));
    morphologyEx(result, result, MORPH_ERODE, kernal);

    cvb::CvBlobs blobs;
    Mat imgLabel,outc;
    cvb::Label(result, imgLabel, blobs);
    int nLaber = blobs.size();

    if(nLaber <= 1)
    {
        morphologyEx(result, result, MORPH_DILATE, kernal);
        return;
    }


    int label = cvb::GetLabel(imgLabel, sz.width/2, sz.height-h_able/2);
    cvFilterByLabel(blobs, label);
    cvb::FilterLabels(imgLabel, outc, blobs);

    Mat kernal2 = getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(17,17));
    morphologyEx(outc, outc, MORPH_DILATE, kernal2);
    result = outc;
}


void TRDetect::getImageMono(cv::Mat &imageMono)
{
    imageMono = imgMonoL.clone();
}
void TRDetect::getImageColor(cv::Mat &imageColor)
{
    imageColor = imgColorL.clone();
}
void TRDetect::getTReigon(cv::Mat &TReigon)
{
    TReigon = result.clone();
}
void TRDetect::getTReigonProb(cv::Mat &TReigonProb)
{
    TReigonProb = resultProb.clone();
}
void TRDetect::getGroundTruth(cv::Mat &groundTruth)
{
    groundTruth = this->groundTruth.clone();
}
Mat TRDetect::getSresult()
{
	//CvFont font;  
	//cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 1.0, 1.0, 0, 2, 8);  
    Mat resultColor;
    RANSAC::cvtMaskn3(result, resultColor);
    Mat sResult = 0.7*imgColorL + 0.3*resultColor;
    //imshow("groundTruth",groundTruth);
	//putText(sResult,"type",Point(20, int(sResult.rows*0.9)), FONT_HERSHEY_PLAIN,sResult.cols/200, cvScalar(0, 0, 200, 0));
    //imshow("result",sResult);
	
	//waitKey(1);
	return sResult;
}
//1111类似于showall中的mask，给不同类别的可行域添加label；
//asphalt(120,120,120),grass(0,240,240),sand(240,240,0);
Mat TRDetect::addLabel(int label)
{
	Mat labelRes;
	labelRes.create(result.size(), CV_8UC3);

	for(int i=0; i<labelRes.rows; ++i)
	{
		const unsigned char *ptr = (unsigned char*)result.ptr(i);
		unsigned char *dst = (unsigned char*)labelRes.ptr(i);
		for(int j=0; j <labelRes.cols; ++j)
		{
			const unsigned char xx = ptr[j];

			if(xx > 205)
			{switch	(label){
case 1://asphalt:gray
				dst[3*j+0] = 120;
				dst[3*j+1] = 120;
				dst[3*j+2] = 120;
				break;
case 2://grass:green
				dst[3*j+0] = 0;
				dst[3*j+1] = 240;
				dst[3*j+2] = 0;
				break;
case 3://sand:yellow
				dst[3*j+0] = 0;
				dst[3*j+1] = 240;
				dst[3*j+2] = 240;
				break;
			}
			}			
			else
			{
				dst[3*j+0] = 0;
				dst[3*j+1] = 0;
				dst[3*j+2] = 0;
			}
		}
	}
	return labelRes;
}

void TRDetect::extract(int label)
{
	//char a[10];
	//string sa;
	//sprintf(a,"%04d", num);
	//sa =a;
	//1st  do the segmentation
	///////////////////////////
	Mat tmp;
    //dataformat transformat
    cvtColor(imgColorL, tmp, CV_BGR2BGRA);
    const unsigned int* buff = (const unsigned int*)tmp.data;

	//only do segmentation once and the number of SP is 180;
    int n=1;
	int& nSuperpixels = layer[n].nSuperpixels;
	// cpu slic
	SLIC mySeg;
	layer[n].imgSeg.create(sz, CV_32S);
	Mat imgSeg = layer[n].imgSeg;
	mySeg.PerformSLICO_ForGivenK( buff, sz.width, sz.height, 
                                      (int*)imgSeg.data,
                                      nSuperpixels,
                                      param.seg.k[n],
                                      param.seg.numItr);

	//2nd compute the center of SP
	//////////////////////////////
	nSuperpixels = layer[n].nSuperpixels;
	assert(nSuperpixels!=0);
	layer[n].info.create(nSuperpixels, INFO_NUM, CV_32S);
	layer[n].flagTR.create(nSuperpixels,1,CV_8U);
	layer[n].input.create(nSuperpixels, param.net.nInputNeurons, CV_32F);
		
	// create references and initialize
	Mat& rimgSeg = layer[n].imgSeg;
	Mat& info   = layer[n].info;
	Mat& input  = layer[n].input;
	Mat& flagTR = layer[n].flagTR;
	info.setTo(Scalar(0));
	input.setTo(Scalar(0.0f));

	// create references and initialize
	const int nBin_H = param.net.nBin_H;
	const int nBin_S = param.net.nBin_S;
	const int nBin_V = param.net.nBin_V;
	const int step1 = nBin_H;
	const int step2 = step1 + nBin_S;
	const int step3 = step2 + nBin_V;
	float scaleH = 180.0f/nBin_H;
	float scaleS = 256.0f/nBin_S;
	float scaleV = 256.0f/nBin_V;

	// get data ready
	for(int r=0; r<sz.height; ++r)
	{
		const auto pSeg = imgSeg.ptr<const unsigned int>(r);
		for(int c=0; c<sz.width; ++c)
		{
			auto label = pSeg[c];
			auto pInfo = (unsigned int *)info.ptr(label);
			++pInfo[INFO_SZ];
			pInfo[INFO_ROW] += r;
			pInfo[INFO_COL] += c;
		}
	}
	// compute the center 
	info.col(INFO_ROW) /= info.col(INFO_SZ);
	info.col(INFO_COL) /= info.col(INFO_SZ);
	Mat iii=result;
	//3rd determine each SP's traversability, 1 for yes, 0 for no
	/////////////////////////////////////////////////////////////
	//indexTR is the index of the traversable SPs
	vector<int> indexTR;
	int* rowK;
	for( int k=0; k < nSuperpixels; k++)
	{
		//???not sure about the position
		rowK = info.ptr<int>(k);
		int r =rowK[1];
		int c =rowK[2];
		if(result.at<uchar>(r,c) > 100){
			flagTR.at<uchar>(k) = 1;
			indexTR.push_back(k);
		}
		else
			flagTR.at<uchar>(k) = 0;
	}
	//如果indexTR的size为0，即没有可行区域识别出来，则中断
	if(indexTR.size()<20)
		return;
	//4th select some SP which is traversable randomly
	////////////////////////////////////////////////////
	//randomTR is the sequence for selected SP's label
	int randomTR[20];
	for(int i=0; i<20; i++){
		srand(unsigned(time(0)));
		randomTR[i] = indexTR[rand()%indexTR.size()];
	}

	//5th extract selected SP's feature
	////////////////////////////////////
	int flag = 0;    //determine whether the SP is selected
	// get data ready
	for(int r=0; r<sz.height; ++r)
	{
		const auto pHSV = imgHSV.ptr<const Point3_<unsigned char>>(r);
		const auto pLBP = imgLBP.ptr<const unsigned char>(r);
		const auto pSeg = imgSeg.ptr<const unsigned int>(r);
		for(int c=0; c<sz.width; ++c)
		{
			auto label = pSeg[c];
			for(int i=0; i<20; i++)
				if (label == randomTR[i])
					flag = 1;

			if (flag==0)
				continue;
			else{
			auto pInput = (float *)input.ptr(label);
			
			int posH = (int)(pHSV[c].x/scaleH);
			int posS = (int)(pHSV[c].y/scaleS) + step1;
			int posV = (int)(pHSV[c].z/scaleV) + step2;
			int posLBP = pLBP[c] + step3;
			
			++pInput[posH];
			++pInput[posS];
			++pInput[posV];
			++pInput[posLBP];
			
			auto pInfo = (unsigned int *)info.ptr(label);
			++pInfo[INFO_SZ];
			pInfo[INFO_ROW] += r;
			pInfo[INFO_COL] += c;
			}
		}
	}
	
	// normalize the network input
	for(int r=0; r<input.rows; ++r)
	{
        input.row(r) /= info.at<uint32_t>(r,INFO_SZ);
	}

	//6th write features into a txt
	//string path = "F:/dataset/part_a/wet-a/camera_stereo_left/frames/";
	///////////////////////////////
	
	//or 直接fp=fopen("feature.txt","w"); 
	FILE*fp=NULL;
	fp=fopen("feature.txt","w"); 
	//fprintf(fp,"\n");
	fclose(fp);
	fp=NULL;

	for(int i=0;i<20;i++)
	{	
		fp=fopen("feature.txt","a");  //创建文件
		fprintf(fp,"%d ",label);
		for(int j=0;j<55;j++){			
			fprintf(fp,"%d:%f ",j+1,input.ptr<float>(randomTR[i])[j]); //从控制台中读入并在文本输出
		}
		fprintf(fp,"\n");
		fclose(fp);
		fp=NULL;//需要指向空，否则会指向原打开文件地址 
	}

}
void TRDetect::simpleExtract(const Mat &img, int num)
{
	//0 get img ready
	/////////////////////
	Mat tmp;
	resize(img,tmp,Size(208,156));
	cvtColor(tmp, imgMonoL, CV_BGR2GRAY);
    cvtColor(tmp, imgHSV, CV_BGR2HSV); // HSV(hue,saturation,value)颜色空间的模型对应于圆柱坐标系中的一个圆锥形子集
    lbpRiu2(imgMonoL, imgLBP);
    //描述局部纹理特征?

	//1st  do the segmentation
	///////////////////////////

    //dataformat transformat
    cvtColor(tmp, tmp, CV_BGR2BGRA);
    const unsigned int* buff = (const unsigned int*)tmp.data;
	
	//only do segmentation once and the number of SP is 180;
    int n=2;
	int& nSuperpixels = layer[n].nSuperpixels;
    //引用
	// cpu slic
	SLIC mySeg;
    //SLIC超像素分割
	layer[n].imgSeg.create(sz, CV_32S);
	Mat imgSeg = layer[n].imgSeg;
	mySeg.PerformSLICO_ForGivenK( buff, sz.width, sz.height, 
                                      (int*)imgSeg.data,
                                      nSuperpixels,
                                      param.seg.k[n],
                                      param.seg.numItr);

	//2nd extract
	//////////////////////////////
	nSuperpixels = layer[n].nSuperpixels;
	assert(nSuperpixels!=0);
	layer[n].info.create(nSuperpixels, INFO_NUM, CV_32S);
	layer[n].input.create(nSuperpixels, param.net.nInputNeurons, CV_32F);
		
	// create references and initialize
	Mat& rimgSeg = layer[n].imgSeg;
	Mat& info   = layer[n].info;
	Mat& input  = layer[n].input;
	info.setTo(Scalar(0));
	input.setTo(Scalar(0.0f));

	// create references and initialize
	const int nBin_H = param.net.nBin_H;
	const int nBin_S = param.net.nBin_S;
	const int nBin_V = param.net.nBin_V;
	const int step1 = nBin_H;
	const int step2 = step1 + nBin_S;
	const int step3 = step2 + nBin_V;
	float scaleH = 180.0f/nBin_H;
	float scaleS = 256.0f/nBin_S;
	float scaleV = 256.0f/nBin_V;
	
	// compute the center 
	info.col(INFO_ROW) /= info.col(INFO_SZ);
	info.col(INFO_COL) /= info.col(INFO_SZ);
	
	// get data ready
	for(int r=0; r<sz.height; ++r)
	{
		const auto pHSV = imgHSV.ptr<const Point3_<unsigned char>>(r);
		const auto pLBP = imgLBP.ptr<const unsigned char>(r);
		const auto pSeg = imgSeg.ptr<const unsigned int>(r);
		for(int c=0; c<sz.width; ++c)
		{
			auto label = pSeg[c];			
			auto pInput = (float *)input.ptr(label);
			
			int posH = (int)(pHSV[c].x/scaleH);
			int posS = (int)(pHSV[c].y/scaleS) + step1;
			int posV = (int)(pHSV[c].z/scaleV) + step2;
			int posLBP = pLBP[c] + step3;
			
			++pInput[posH];
			++pInput[posS];
			++pInput[posV];
			++pInput[posLBP];
				
			auto pInfo = (unsigned int *)info.ptr(label);
			++pInfo[INFO_SZ];
			pInfo[INFO_ROW] += r;
			pInfo[INFO_COL] += c;
			
		}
	}
	
	// normalize the network input
	for(int r=0; r<input.rows; ++r)
	{
        input.row(r) /= info.at<uint32_t>(r,INFO_SZ);
	}

	//6th write features into a txt
	///////////////////////////////
	for(int i=0;i<nSuperpixels;i++)
	{
		FILE*fp=NULL;//需要注意
		//文件名是变量
		char filename[30]={0};
        string file = "g"+to_string(num)+".txt";
		const char* chfile = file.c_str();
		strcpy(filename,chfile);
		
        fp=fopen(filename,"a");  //创建文件
		//if(NULL==fp) return -1;//要返回错误代码
		//while(scanf("%c",&c)!=EOF)
		//add the label first, 1 for asphalt, 2 for grass, 3 for sand
        fprintf(fp,"%d ",2);
		for(int j=0;j<55;j++){			
			
			fprintf(fp,"%d:",j+1); //从控制台中读入并在文本输出
			fprintf(fp,"%f ",input.ptr<float>(i)[j]);
		}
		fprintf(fp,"\n");
		fclose(fp);
		fp=NULL;//需要指向空，否则会指向原打开文件地址 
	}
}

