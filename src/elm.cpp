// ref1: http://dovgalecs.com/blog/extreme-learning-machine-matlab-mex-implementation/
// ref2: http://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html

#include "elm.h"

#include <iostream>
#include <set>
#include "opencv2/gpu/gpu.hpp"
#include "helper_timer.h"
#include "Eigen/Dense"

using namespace cv;
using namespace Eigen;

#define MatrixF    Matrix<float, Dynamic, Dynamic, RowMajor>
#define VectorF    Matrix<float, Dynamic, 1>
#define RowVectorF Matrix<float, 1, Dynamic>
#define DiagF      DiagonalMatrix<float, Dynamic>

// n - number of input nodes
// L - number of hidden nodes
// N - number of input data
// m - number of output nodes
// inputWeight  : n*L matrix
// bias         : 1*L matrix
// input        : N*n matrix
// output       : N*m matrix
// outputWeight : L*m matrix
// H = g( input * inputWeight + extBias) : N*L matrix
// output = H * outputWeight
ELM::ELM(int nInputNeurons, int nHiddenNeurons, int elm_af):
nInputNeurons(nInputNeurons),
nHiddenNeurons(nHiddenNeurons),
activatFunction(elm_af)
{
    RNG rng(getTickCount());

    inputWeight.create(nInputNeurons, nHiddenNeurons, CV_32F);
    rng.fill(inputWeight, RNG::UNIFORM, -0.9999999,1);

    bias.create(1, nHiddenNeurons, CV_32F);
    rng.fill(bias,RNG::UNIFORM,0.0000001,1);
}

ELM::~ELM()
{

}

#ifdef __CUDACC__
// TODO: add GPU code here
void train(const cv::Mat& input, const cv::Mat& output, cv::Mat& outputWeight, const cv::Mat& weight)
{

}
// bool elm_train(const Mat& train_data_o,const Mat& ground_truth_o,const Mat & weight_o, Mat &InputWeight,Mat &Bias,Mat &OutputWeight){

//     StopWatchInterface *timer = NULL;
//     sdkCreateTimer(&timer);
//     sdkStartTimer(&timer);

//     Mat ground_truth=ground_truth_o.t();//将矩阵转置
//     Mat ground_truth_reverse=-1*ground_truth;
//     ground_truth.push_back(ground_truth_reverse);
//     ground_truth_reverse.release();
//     ground_truth.convertTo(ground_truth,CV_32F);

//     //printf("   file %s line %d time spent: %.2fms\n", __FILE__, __LINE__, sdkGetTimerValue(&timer));

//     gpu::GpuMat InputWeight_d(InputWeight);
//     gpu::GpuMat train_data_d(train_data_o);
//     gpu::GpuMat temp_h_d;

//     Mat tempH;
//     /*Mat train_data=train_data_o.t();*/
//     //Mat tempH=InputWeight*train_data; //InputWeight is N*M  
//     //train_data.release(); //  Release input of training data 
//     gpu::gemm(InputWeight_d,train_data_d,1.0,gpu::GpuMat(),0,temp_h_d,GEMM_2_T);

//     //printf("   file %s line %d GEMM time spent: %.2fms\n", __FILE__, __LINE__, sdkGetTimerValue(&timer));

//     //tempH=tempH+BiasMatrix;
//     int NumberofTrainingData=train_data_o.rows;
//     int NumberofInputNeurons=train_data_o.cols;
//     Mat BiasMatrix=Bias*Mat::ones(1,NumberofTrainingData,CV_32F);  // Extend the bias matrix BiasofHiddenNeurons to match the demention of H
//     gpu::add(temp_h_d,gpu::GpuMat(BiasMatrix),temp_h_d);
//     //printf("   file %s line %d tempH=tempH+BiasMatrix time spent: %.2fms\n", __FILE__, __LINE__, sdkGetTimerValue(&timer));

//     // 1/[1+e^(-z)]
//     gpu::multiply(temp_h_d,-1,temp_h_d);
//     gpu::exp(temp_h_d,temp_h_d); //the exp of  original matrix
//     gpu::add(temp_h_d,Scalar(1.0f),temp_h_d);
//     gpu::divide(1.0f,temp_h_d,temp_h_d);

//     //temp_h_d.download(tempH);

//     //printf("   file %s line %d SIGMOD time spent: %.2fms\n", __FILE__, __LINE__, sdkGetTimerValue(&timer));

//     Mat weight = weight_o.t();
//     Mat weight_ext(temp_h_d.rows,weight.cols,weight.type());
//     for(int i =0; i<weight_ext.rows; i++)
//         weight.row(0).copyTo(weight_ext.row(i));

//     Mat common_matrix;
//     gpu::GpuMat commatrix_d;
//     gpu::GpuMat weight_ext_d(weight_ext);
//     gpu::multiply(temp_h_d,weight_ext_d,commatrix_d);

//     //printf("   file %s line %d time spent: %.2fms\n", __FILE__, __LINE__, sdkGetTimerValue(&timer));
//     gpu::GpuMat OutputWeight_d;
//     gpu::gemm(commatrix_d,temp_h_d,1.0,gpu::GpuMat(),0,OutputWeight_d,GEMM_2_T);
    
//     OutputWeight_d.download(OutputWeight);
//     for(int i=0;i<OutputWeight.cols;++i){
//         OutputWeight.at<float>(i,i)+=1.5;
//     }
//     OutputWeight=OutputWeight.inv();

//     //printf("   file %s line %d time spent: %.2fms\n", __FILE__, __LINE__, sdkGetTimerValue(&timer));
//     Mat temp_t;
//     gpu::GpuMat temp_d;
//     gpu::GpuMat ground_truth_d(ground_truth);
//     gpu::gemm(commatrix_d,ground_truth_d,1.0,gpu::GpuMat(),0,temp_d,GEMM_2_T);
//     temp_d.download(temp_t);

//     //printf("   file %s line %d time spent: %.2fms\n", __FILE__, __LINE__, sdkGetTimerValue(&timer));

//     OutputWeight=OutputWeight*temp_t;

//     printf("    file %s line %d time spent: %.2fms\n", __FILE__, __LINE__, sdkGetTimerValue(&timer));
//     sdkDeleteTimer(&timer);
//     return true;
// }
#else
// n - number of input nodes
// L - number of hidden nodes
// N - number of input data
// m - number of output nodes
// input        : N*n matrix
// inputWeight  : n*L matrix
// bias         : 1*L matrix
// output       : N*m matrix
// outputWeight : L*m matrix
// weight       : N*1 matrix
// H = g(input * inputWeight + extBias) : N*L matrix
// output = H * outputWeight
// outputWeight = (Ht*H + I/C)^-1 * Ht*output
// outputWeight = (Ht*weight*H + I/C)^-1 * Ht*weight*output
void ELM::train(const cv::Mat& input, const cv::Mat& output, cv::Mat& outputWeight, const cv::Mat& weight)
{
    // StopWatchInterface *timer = NULL;
    // sdkCreateTimer(&timer);
    // sdkStartTimer(&timer);
    float c=1.5f; // TODO

    int L = inputWeight.cols;
    int N = input.rows;
    int m = output.cols;

    outputWeight.create(L, m, CV_32F);

    // Map OpenCV Mat to Eigen Matrix, no copy data, just map memory
    Map<const MatrixF> inputE((float *)input.data, input.rows, input.cols);
    Map<const MatrixF> inputWeightE((float *)inputWeight.data, inputWeight.rows, inputWeight.cols);
    Map<const RowVectorF> biasE((float *)bias.data, bias.rows, bias.cols);
    Map<MatrixF> outputWeightE((float *)outputWeight.data, outputWeight.rows, outputWeight.cols);
    Map<const MatrixF> outputE((float *)output.data, output.rows, output.cols);

    MatrixF I = MatrixXf::Identity(L, L);

    DiagF weightE;
    if(!weight.empty())
    {
        Map<const VectorF> _weightE((float *)weight.data, weight.rows, weight.cols);
        weightE.diagonal() = _weightE;
    }
    else
    {
        weightE.setIdentity(N); // weightE = I
    }

    // tempH = input * inputWeight + extBias
    MatrixF H = (inputE * inputWeightE).rowwise() + biasE;

    // H = g(tempH)
    switch (activatFunction)
    {
        case ELM_AF_SIGMOID:
            // 1/(1+e^(-x))
            H = 1/(1+exp(-H.array()));  break;

        case ELM_AF_SIGMOID_FAST:
            // x/(1+|x|)
            H = (0.5f*H.array())/(1+H.array().abs())+0.5f; break;

        default:
            H = (0.5f*H.array())/(1+H.array().abs())+0.5f;
    }
    
    // outputWeight = (Ht*weight*H + I/C)^-1 * Ht*weight*output
    // outputWeight = A^-1*B
    // A * outputWeight = B
    MatrixF HtW = H.transpose() * weightE;
    MatrixF A = (HtW*H + I*c); // c = 1/C
    MatrixF B = HtW * outputE;

    // TODO: select a mothed to compute outputWeight
    outputWeightE = A.llt().solve(B);           // Cholesky decomposition, A is Positive definite 
    // outputWeightE = A.partialPivLu().solve(B);  // LU decomposition, A is Invertible 
    // outputWeightE = A.inverse()*B;              // FullPivLU

    // printf("   file %s line %d time spent: %.2fms\n", __FILE__, __LINE__, sdkGetTimerValue(&timer));
}
#endif

// n - number of input nodes
// L - number of hidden nodes
// N - number of input data
// m - number of output nodes
// input        : N*n matrix
// inputWeight  : n*L matrix
// bias         : 1*L matrix
// output       : N*m matrix
// outputWeight : L*m matrix
// H = g(input * inputWeight + extBias) : N*L matrix
// output = H * outputWeight
void ELM::predict(const cv::Mat& input, cv::Mat& output, const cv::Mat& outputWeight)
{
    //StopWatchInterface *timer = NULL;
    //sdkCreateTimer(&timer);
    //sdkStartTimer(&timer);
    
    int N = input.rows;
    int m = outputWeight.cols;

    output.create(N, m, CV_32F);

    // Map OpenCV Mat to Eigen Matrix, no copy data, just map memory
    Map<const MatrixF> inputE((float *)input.data, input.rows, input.cols);
    Map<const MatrixF> inputWeightE((float *)inputWeight.data, inputWeight.rows, inputWeight.cols);
    Map<const RowVectorF> biasE((float *)bias.data, bias.rows, bias.cols);
    Map<const MatrixF> outputWeightE((float *)outputWeight.data, outputWeight.rows, outputWeight.cols);
    Map<MatrixF> outputE((float *)output.data, output.rows, output.cols);

    // tempH = input * inputWeight + extBias
    MatrixF H = (inputE * inputWeightE).rowwise() + biasE;

    // H = g(tempH)
    switch (activatFunction)
    {
        case ELM_AF_SIGMOID:
            // 1/(1+e^(-x))
            H = 1/(1+exp(-H.array()));  break;

        case ELM_AF_SIGMOID_FAST:
            // x/(1+|x|)
            H = (0.5f*H.array())/(1+H.array().abs())+0.5f; break;

        default:
            H = (0.5f*H.array())/(1+H.array().abs())+0.5f;
    }

    // output = H * outputWeight
    outputE = H * outputWeightE;

    //printf("   file %s line %d time spent: %.2fms\n", __FILE__, __LINE__, sdkGetTimerValue(&timer));  
}

// n - number of input nodes
// L - number of hidden nodes
// N - number of input data
// m - number of output nodes
// laber        : N*1 matrix
// target       : N*m matrix
// note: laber should contain [0,nLaber-1]
void ELM::buildTargetMat(const cv::Mat& laber, cv::Mat& target, int nLaber)
{
    int N = laber.rows;
    int m = nLaber;

    target.create(N, m, CV_32F);
    target.setTo(Scalar(-1.0f));

    int index;
    float* pSrc = (float *)laber.data;
    for(int r=0; r<N; ++r)
    {
        index = (int)(*pSrc);
    	float* pDst = target.ptr<float>(r);
    	pDst[index] = 1.0f;
        pSrc++;
    }
}