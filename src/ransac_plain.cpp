#include <iostream>
#include <stdio.h>
#include <time.h>
#include "Eigen/Dense"
#include <opencv2/opencv.hpp>

namespace RANSAC
{
    using namespace std;
    using namespace cv;

    // Parallel Loop Body to determine whether the point is on the plane
    class Parallel_Cos: public ParallelLoopBody
    {
    public:
    Parallel_Cos(const Mat &_img, Mat &_mask1, Mat &_mask2, Eigen::Vector4d &_model, double _threshold) : img(_img), mask1(_mask1), mask2(_mask2), model(_model)
    {
        threshold = _threshold;
    }

    void operator() (const Range &r) const
    {
        for(int j=r.start; j<r.end; ++j)
        {
            Point3f *current = (Point3f *)img.ptr(j);
            Point3f *last = current + img.cols;
            unsigned char *dst_in = (unsigned char *)mask1.ptr(j);
            unsigned char *dst_out = (unsigned char *)mask2.ptr(j);
            for (; current != last; ++current,++dst_in,++dst_out)
            {
                const Point3f p = *current;
                if(p.x!=0.0f && p.y!=0.0f && p.z!=0.0f)
                {
                    if( abs(model[0]*p.x+model[1]*p.y+model[2]*p.z+model[3]) < threshold )
                        *dst_in = 0xFF; // on the plane
                    else
                        *dst_out = 0x7F; // not on the plane
                }
            }
        }
    }

    private:
        const Mat img;
        Mat mask1;
        Mat mask2;
        Eigen::Vector4d model;
        double threshold;
    };
    
    // ransacForPlane
    template <typename T>
    static bool ransacForPlane(
                const std::vector<T>  &x,
                const std::vector<T>  &y,
                const std::vector<T>  &z,
                const double          distanceThreshold,
                const unsigned int    minimumSizeSamplesToFit,
                std::vector<int>      &out_best_inliers,
                std::vector<double>   &out_best_model,
                const size_t          min_inliers_for_valid_plane,
                const double          p,
                const size_t          maxIter = 1000,
                bool                  verbose = false )
    {
        const size_t Npts = x.size();
        if (verbose)
            printf("[RANSAC] number of points:%d\n", (int)Npts);

        out_best_model.clear();
        out_best_inliers.clear();

        size_t trialcount = 0;
        size_t bestscore = std::string::npos; // npos will mean "none"
        size_t N = 1;     // Dummy initialisation for number of trials.

        int ind[3]={0};
        double coefs[4]={0.0};
        srand((unsigned)time(0));

        // get model
        while (N > trialcount)
        {
            bool degenerate=true;
            while (degenerate)
            {
                // Generate s random indicies in the range 1..npts
                ind[0] = rand()%Npts;
                ind[1] = rand()%Npts;
                ind[2] = rand()%Npts;
                Point3d p1 = Point3d(x[ind[0]], y[ind[0]], z[ind[0]]);
                Point3d p2 = Point3d(x[ind[1]], y[ind[1]], z[ind[1]]);
                Point3d p3 = Point3d(x[ind[2]], y[ind[2]], z[ind[2]]);

                double dx1=p2.x-p1.x;
                double dy1=p2.y-p1.y;
                double dz1=p2.z-p1.z;
                double dx2=p3.x-p1.x;
                double dy2=p3.y-p1.y;
                double dz2=p3.z-p1.z;
                coefs[0]=dy1*dz2-dy2*dz1;
                coefs[1]=dz1*dx2-dz2*dx1;
                coefs[2]=dx1*dy2-dx2*dy1;
                if (abs(coefs[0])<0.00001&&abs(coefs[1])<0.00001&&abs(coefs[2])<0.00001)
                {
                    degenerate = true;  // regenerate a model
                }
                else
                {
                    degenerate = false;
                    coefs[3]=-coefs[0]*p1.x-coefs[1]*p1.y-coefs[2]*p1.z;
                }
            }

            // get inliers
            vector<int> inliers;
            inliers.reserve(100);
            const double num = 1/sqrt(coefs[0]*coefs[0]+coefs[1]*coefs[1]+coefs[2]*coefs[2]);
            for (size_t i=0;i<Npts;i++)
            {
                const double d = abs(coefs[0]*x[i]+coefs[1]*y[i]+coefs[2]*z[i]+coefs[3]) * num;
                if (d<distanceThreshold)
                    inliers.push_back(i);
            }

            // Find the number of inliers to this model.
            const size_t ninliers = inliers.size();
            bool update_estim_num_iters = (trialcount==0); // Always update on the first iteration, regardless of the result (even for ninliers=0)

            if (ninliers > bestscore || (bestscore==std::string::npos && ninliers!=0))
            {
                bestscore = ninliers;  // Record data for this model

                out_best_model    = std::vector<double>(coefs, coefs + sizeof(coefs)/sizeof(double));
                out_best_inliers  = inliers;
                update_estim_num_iters = true;
            }

            if (update_estim_num_iters)
            {
                // Update estimate of N, the number of trials to ensure we pick,
                // with probability p, a data set with no outliers.
                double fracinliers =  ninliers/static_cast<double>(Npts);
                double pNoOutliers = 1 -  pow(fracinliers,static_cast<double>(minimumSizeSamplesToFit));

                pNoOutliers = std::max( std::numeric_limits<double>::epsilon(), pNoOutliers);  // Avoid division by -Inf
                pNoOutliers = std::min(1.0 - std::numeric_limits<double>::epsilon() , pNoOutliers); // Avoid division by 0.
                N = (size_t)(log(1-p)/log(pNoOutliers));
                if (verbose)
                    printf("[RANSAC] Iter #%u Estimated number of iters: %u  pNoOutliers = %f  #inliers: %u\n", (unsigned)trialcount ,(unsigned)N,pNoOutliers, (unsigned)ninliers);
            }

            ++trialcount;

            if (verbose)
                printf("[RANSAC] trial %u out of %u \r",(unsigned int)trialcount, (unsigned int)ceil(static_cast<double>(N)));

            // Safeguard against being stuck in this loop forever
            if (trialcount > maxIter)
            {
                if (verbose)
                    printf("[RANSAC] Warning: maximum number of trials (%u) reached\n", (unsigned)maxIter);
                break;
            }
        }

        if (out_best_inliers.size() > min_inliers_for_valid_plane)
        {  // We got a solution
            if (verbose)
                printf("[RANSAC]  Finished in %u iterations.\n",(unsigned)trialcount );
            return true;
        }
        else
        {
            if (verbose)
                printf("[RANSAC] Warning: Finished without any proper solution.\n");
            return false;
        }
    }


    Vec4d getMonoMask(
        const Mat &pcd,
        Mat &mask,
        Size sz,
        const double threshold,
        const int stride,
        const float validRate,
        const int numResample
        )
    {
        Vec4d m;
        mask.create(sz, CV_8U);

        // Load valid data into vector
        // ------------------------------------
        std::vector<float> x,y,z;
        x.reserve(0xFF);
        y.reserve(0xFF);
        z.reserve(0xFF);
        for(int i=0; i<pcd.rows; i=i+stride)
        {
            Point3f *ptr = (Point3f* )pcd.ptr(i);
            for(int j=0; j <pcd.cols; j=j+stride)
            {
                const Point3f p = ptr[j];
                // set valid range for X,Y,Z
                // in OpenCV reprojectImageTo3D, if point is invalid, Z=10000
                if(p.z>0.0f && p.z<5000.0f)
                {
                    x.push_back(p.x);
                    y.push_back(p.y);
                    z.push_back(p.z);
                }
            }
        }

        if (x.empty())
        {
            mask.setTo(Scalar(0));
            return m;
        }

        // Run RANSAC
        // ------------------------------------
        std::vector<int> best_inliers;
        std::vector<double> this_best_model(4);
        const size_t min_inliers_for_valid_plane = (size_t)(x.size()*validRate);

        bool result = ransacForPlane(
            x,y,z,
            threshold,
            3,  // Minimum set of points
            best_inliers,
            this_best_model,
            min_inliers_for_valid_plane,
            0.999,  // Prob. of good result
            500, // max trial
            //true // Verbose
            false
            );

        if (result != true)
        {
            mask.setTo(Scalar(0));
            return m;
        }

        // Recalculate the model using LSQ
        // ------------------------------------
        const int ninliers = best_inliers.size();

        int nResample = numResample < ninliers ? numResample:ninliers;
        Eigen::Matrix<double, Eigen::Dynamic, 3> rsMat;
        rsMat.resize(nResample, Eigen::NoChange);

        srand((unsigned int) time(0));
        for( int i=0; i<nResample;i++)
        {
            int idx = rand()%ninliers;
            rsMat.row(i) = Eigen::Vector3d(x[best_inliers[idx]], y[best_inliers[idx]], z[best_inliers[idx]]);
        }

        Eigen::RowVector3d rsMean = rsMat.colwise().mean();
        //cout << "rsMean: "<< rsMean <<endl;
        
        rsMat.rowwise() -= rsMean;

        Eigen:: JacobiSVD<Eigen::MatrixXd> svd(rsMat, Eigen::ComputeThinV);
        //cout << "Its singular values are:" << endl << svd.singularValues() << endl;
        //cout << "Its right singular vectors are the columns of the thin V matrix:" << endl << svd.matrixV() << endl;

        Eigen::Vector3d normVec = svd.matrixV().col(2); // get the last row

        Eigen::Vector4d model; // ax+by+cz+d=0
        //model<< normVec, -model[0]*rsMean[0]-model[1]*rsMean[1]-model[2]*rsMean[2];
        model<< normVec, -normVec.dot(rsMean);
        //cout << model <<endl;

        // after SVD, the vector V is already a norm vector
        // double s = normVec.norm();
        // cout<<"scale" << s<<endl;
        // model /= s;
        // cout << "after "<< model <<endl;

        // get mask
        // ------------------------------------
        Mat mask_inliers(pcd.size(),CV_8UC1,Scalar(0));
        Mat mask_outliers(pcd.size(),CV_8UC1,Scalar(0));

        // parallel_for_( Range(0,mask.rows) , Parallel_Cos(pcd,mask_inliers,mask_outliers,model,threshold)) ;
        for(int i=0; i<pcd.rows; ++i)
        {
            Point3f *ptr = (Point3f* )pcd.ptr(i);
            unsigned char *dst_in = (unsigned char *)mask_inliers.ptr(i);
            unsigned char *dst_out = (unsigned char *)mask_outliers.ptr(i);
            for(int j=0; j <pcd.cols; ++j)
            {
                const Point3f p = ptr[j];
                if(p.z>0.0f && p.z<5000.0f)
                {
                    if( abs(model[0]*p.x+model[1]*p.y+model[2]*p.z+model[3]) < threshold )
                        dst_in[j] = 0xFF; // on the plane
                    else
                        dst_out[j] = 0x7F; // not on the plane
                }
            }
        }

        resize(mask_inliers,mask_inliers,sz);
        resize(mask_outliers,mask_outliers,sz);

        // do a blur first to reduce noise
        const int blurSize = (int)(sz.width/45.0f);
        blur(mask_inliers,mask_inliers,Size(blurSize,blurSize));
        blur(mask_outliers,mask_outliers,Size(blurSize,blurSize));

        // kick out noise
        cv::threshold(mask_inliers,mask_inliers,(0xFF-85),0xFF,THRESH_BINARY);
        cv::threshold(mask_outliers,mask_outliers,(0x7F-85),0x7F,THRESH_BINARY);

        mask = mask_inliers + mask_outliers;

        const int kernalSize = (int)(sz.width/20.0f + 0.5f);
        Mat kernal = getStructuringElement(MORPH_ELLIPSE,Size(kernalSize,kernalSize));
        morphologyEx(mask,mask,MORPH_CLOSE,kernal);
		morphologyEx(mask,mask,MORPH_OPEN,kernal);
		
        m[0] = model[0];
        m[1] = model[1];
        m[2] = model[2];
        m[3] = model[3];
        return m;
    }

    void cvtMaskn3(const Mat &mono, Mat &mask)
    {
        mask.create(mono.size(), CV_8UC3);

        for(int i=0; i<mask.rows; ++i)
        {
            const unsigned char *ptr = (unsigned char*)mono.ptr(i);
            unsigned char *dst = (unsigned char*)mask.ptr(i);
            for(int j=0; j <mask.cols; ++j)
            {
                const unsigned char xx = ptr[j];

                if(xx > 205)
                {
                    dst[3*j+0] = 0;
                    dst[3*j+1] = 255;
                    dst[3*j+2] = 0;
                }
                else if(xx>75 && xx<175)
                {
                    dst[3*j+0] = 0;
                    dst[3*j+1] = 0;
                    dst[3*j+2] = 255;
                }
                else
                {
                    dst[3*j+0] = 0;
                    dst[3*j+1] = 0;
                    dst[3*j+2] = 0;
                }
            }
        }
    }
}