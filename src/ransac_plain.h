#ifndef RANSAC_PLAIN_H
#define RANSAC_PLAIN_H

namespace RANSAC
{
    using namespace cv;
    Vec4d getMonoMask(
        const Mat &pcd,
        Mat &mask,
        Size sz,
        const double threshold,
        const int stride = 3,
        const float validRate = 0.3f,
        const int numResample = 200
        );

    void cvtMaskn3(const Mat &mono, Mat &mask);
}

#endif
