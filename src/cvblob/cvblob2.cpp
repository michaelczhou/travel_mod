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

#include "cvblob2.h"

namespace cvb
{
    //cvblob.cpp

    void RenderBlob(Mat imgLabel, CvBlob *blob, Mat imgSource, Mat& imgDest, unsigned short mode, CvScalar const &color, double alpha)
    {
        IplImage tempLabel=imgLabel;
        IplImage img=imgSource;
        IplImage *imgOut = cvCreateImage(cvGetSize(&img), IPL_DEPTH_8U, 3);
        cvZero(imgOut);
        cvRenderBlob(&tempLabel,blob,&img,imgOut,mode,color,alpha);
        imgDest=cvarrToMat(imgOut);
    }

    void RenderBlobs(Mat imgLabel, CvBlobs &blobs, Mat imgSource, Mat& imgDest, unsigned short mode, double alpha)
    {
        IplImage tempLabel=imgLabel;
        IplImage img=imgSource;
        IplImage *imgOut = cvCreateImage(cvGetSize(&img), IPL_DEPTH_8U, 3);
        cvZero(imgOut);
        cvRenderBlobs(&tempLabel,blobs,&img,imgOut,mode,alpha);
        imgDest=cvarrToMat(imgOut);
    }

    void SaveImageBlob(const char *filename, Mat img, CvBlob const *blob)
    {
        IplImage temp=img;
        cvSaveImageBlob(filename,&temp,blob);
    }



    //cvlabel.cpp
    unsigned int Label (Mat src, Mat& imgOut, CvBlobs &blobs)
    {
        IplImage temp=src;
        IplImage *labelImg = cvCreateImage(cvGetSize(&temp),IPL_DEPTH_LABEL,1);
        unsigned int numPixels=cvLabel(&temp, labelImg, blobs);
        imgOut=cvarrToMat(labelImg);
        return numPixels;
    }


    void FilterLabels(Mat imgLabel, Mat& imgOut, const CvBlobs &blobs)
    {
        IplImage temp=imgLabel;
        IplImage *out = cvCreateImage(cvGetSize(&temp),IPL_DEPTH_8U,1);
        cvFilterLabels(&temp, out,blobs);
        imgOut=cvarrToMat(out);
    }

    CvLabel GetLabel(Mat img, unsigned int x, unsigned int y)
    {
        IplImage temp=img;
        return cvGetLabel(&temp,x,y);
    }

    // cvcolor.cpp
    Scalar BlobMeanColor(CvBlob const *blob, Mat imgLabel, Mat img)
    {
        IplImage tempLabel=imgLabel;
        IplImage source=img;
        CvScalar color=cvBlobMeanColor(blob,&tempLabel,&source);
        return Scalar(color.val[0],color.val[1],color.val[2]);

    }

    // cvcontour.cpp
    void RenderContourChainCode(CvContourChainCode const *contour, Mat& img, CvScalar const &color)
    {
        IplImage temp=img;
        cvRenderContourChainCode(contour,&temp,color);
    }

    void RenderContourPolygon(CvContourPolygon const *contour, Mat& img, CvScalar const &color)
    {
        IplImage temp=img;
        cvRenderContourPolygon(contour,&temp,color);
    }

    // cvtrack.cpp
    void RenderTracks(CvTracks const tracks, Mat imgSource, Mat& imgDest, unsigned short mode, CvFont *font)
    {
        IplImage imgS=imgSource;
        IplImage imgD=imgDest;
        cvRenderTracks(tracks,&imgS,&imgD,mode,font);
    }


}
