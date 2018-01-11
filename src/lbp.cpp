#include "lbp.h"
using namespace cv;

int map_x[]={-1, -1, -1,  0,  1,  1,  1,  0};
int map_y[]={-1,  0,  1,  1,  1,  0, -1, -1};

// ref:http://www.cse.oulu.fi/CMV/Downloads/LBPMatlab   
//     generate from getmapping.m
// riu2 -p8 -r1
unsigned char table_riu2[256]={    
    0, 1, 1, 2, 1, 9, 2, 3, 1, 9, 9, 9, 2, 9, 3, 4, 
    1, 9, 9, 9, 9, 9, 9, 9, 2, 9, 9, 9, 3, 9, 4, 5, 
    1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 
    2, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 4, 9, 5, 6, 
    1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 
    2, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 
    3, 9, 9, 9, 9, 9, 9, 9, 4, 9, 9, 9, 5, 9, 6, 7, 
    1, 2, 9, 3, 9, 9, 9, 4, 9, 9, 9, 9, 9, 9, 9, 5, 
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6, 
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 
    2, 3, 9, 4, 9, 9, 9, 5, 9, 9, 9, 9, 9, 9, 9, 6, 
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 
    3, 4, 9, 5, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 7, 
    4, 5, 9, 6, 9, 9, 9, 7, 5, 6, 9, 7, 6, 7, 7, 8, 
};


void lbpRiu2(const Mat& imgGrey, Mat& imgLBP)
{
    int radius = 1;
    int neighbors = 8;
    imgLBP.create(imgGrey.size(), CV_8U);
    imgLBP.setTo(Scalar(0));

    for(int n=0; n<neighbors; n++)
    {
        int dx = map_x[n];
        int dy = map_y[n];
        for(int i=radius; i<imgGrey.rows-radius; ++i)
        { 
            const unsigned char* src = imgGrey.ptr<const unsigned char>(i);
            const unsigned char* cmp = imgGrey.ptr<const unsigned char>(i+dx);
            unsigned char* dst = imgLBP.ptr<unsigned char>(i);

            for(int j=radius; j<imgGrey.cols-radius; ++j)
            {
                dst[j] |= ((cmp[j+dy]>=src[j])<<n);
                if(n == (neighbors-1))
                    dst[j] = table_riu2[dst[j]];
            }
        }
    }

    // deal with border
    imgLBP.row(1).copyTo(imgLBP.row(0));
    imgLBP.row(imgLBP.rows-2).copyTo(imgLBP.row(imgLBP.rows-1));
    imgLBP.col(1).copyTo(imgLBP.col(0));
    imgLBP.col(imgLBP.cols-2).copyTo(imgLBP.col(imgLBP.cols-1));
}
