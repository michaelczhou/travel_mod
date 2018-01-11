// Copyright (C) 2012 by BlueKid
// bluekid70@gmail.com
//
// This file is contribution for cvBlob. cvBlob2 interface for the C++ API for Opencv 2.x.
//
// cvBlob is free software: you can redistribute it and/or modify
// it under the terms of the Lesser GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// cvBlob is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// Lesser GNU General Public License for more details.
//
// You should have received a copy of the Lesser GNU General Public License
// along with cvBlob.  If not, see <http://www.gnu.org/licenses/>.
//

/// \file cvblob2.h
/// \brief OpenCV Blob2 header file.
#ifndef CVBLOB2_H
#define CVBLOB2_H
#include "opencv2/imgproc/imgproc.hpp"
#include "cvblob.h"

namespace cvb
{
    using namespace cv;
    // cvblob.cpp

    void RenderBlob(Mat imgLabel, CvBlob *blobs, Mat imgSource, Mat& imgDest, unsigned short mode=0x000f, CvScalar const &color=CV_RGB(255, 255, 255), double alpha=1.);
    void RenderBlobs(Mat imgLabel, CvBlobs &blobs, Mat imgSource, Mat& imgDest, unsigned short mode=0x000f, double alpha=1.);
    void SaveImageBlob(const char *filename, Mat img, CvBlob const *blob);

    // cvlabel.cpp
    unsigned int Label (Mat src, Mat& imgOut, CvBlobs &blobs);
    void FilterLabels(Mat imgLabel, Mat& imgOut, const CvBlobs &blobs);
    CvLabel GetLabel(Mat img, unsigned int x, unsigned int y);

    // cvcolor.cpp
    Scalar BlobMeanColor(CvBlob const *blob, Mat imgLabel, Mat img);

    // cvcontour.cpp
    void RenderContourChainCode(CvContourChainCode const *contour, Mat& img, CvScalar const &color=CV_RGB(255, 255, 255));
    void RenderContourPolygon(CvContourPolygon const *contour, Mat& img, CvScalar const &color);

    // cvtrack.cpp
    void RenderTracks(CvTracks const tracks,Mat imgSource, Mat& imgDest, unsigned short mode=0x000f, CvFont *font=NULL);

}


#endif // CVBLOB2_H
