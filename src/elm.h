#ifndef ELM_H
#define ELM_H
#include "opencv2/core/core.hpp"
class ELM
{
public:
    enum
    {
        ELM_AF_SIGMOID,
        ELM_AF_SIGMOID_FAST,
    };

    ELM(int nInputNeurons, int nHiddenNeurons, int elm_af);
    ~ELM();

    // if weight is not empty, use weight ELM
    void train(const cv::Mat& input, const cv::Mat& output, cv::Mat& outputWeight, const cv::Mat& weight);
    void predict(const cv::Mat& input, cv::Mat& output, const cv::Mat& outputWeight);
    void buildTargetMat(const cv::Mat& laber, cv::Mat& target, int nLaber);

private:
    int nInputNeurons;
    int nHiddenNeurons;
    int activatFunction;
    cv::Mat inputWeight;
    cv::Mat bias;
};
#endif
