#include "cudaImgTrans.h"
#include "cudaUtil.h"


__host__ void Rgb2CIELab( uchar4* inputImg, float4* outputImg, int width, int height )
{
	dim3 ThreadPerBlock(BLCK_SIZE,BLCK_SIZE);
	dim3 BlockPerGrid(iDivUp(width,BLCK_SIZE),iDivUp(height,BLCK_SIZE));
	kRgb2CIELab<<<BlockPerGrid,ThreadPerBlock>>>(inputImg,outputImg,width,height);

}

__global__ void kRgb2CIELab(uchar4* inputImg, float4* outputImg, int width, int height)
{
	int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
	int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

    if(offset>=width*height)
        return;

	uchar4 nPixel=inputImg[offset];
	
	float _b=(float)nPixel.x/255.0f;
	float _g=(float)nPixel.y/255.0f;
	float _r=(float)nPixel.z/255.0f;

	float x=_r*0.412453f	+_g*0.357580f	+_b*0.180423f;
	float y=_r*0.212671f	+_g*0.715160f	+_b*0.072169f;
	float z=_r*0.019334f	+_g*0.119193f	+_b*0.950227f;

	x/=0.950456f;
	float y3=exp(log(y)/3.0f);
	z/=1.088754f;

	float l,a,b;

	x = x>0.008856f ? exp(log(x)/3.0f) : (7.787f*x+0.13793f);
	y = y>0.008856f ? y3 : 7.787f*y+0.13793f;
	z = z>0.008856f ? z/=exp(log(z)/3.0f) : (7.787f*z+0.13793f);
	
	l = y>0.008856f ? (116.0*y3-16.0) : 903.3f*y;
	a=(x-y)*500.0f;
	b=(y-z)*200.0f;
	
	float4 fPixel;
	fPixel.x=l;
	fPixel.y=a;
	fPixel.z=b;

	outputImg[offset]=fPixel;
}


__host__ void Rgb2XYZ( uchar4* inputImg, float4* outputImg, int width, int height )
{
	dim3 ThreadPerBlock(BLCK_SIZE,BLCK_SIZE);
	dim3 BlockPerGrid(iDivUp(width,BLCK_SIZE),iDivUp(height,BLCK_SIZE));
	kRgb2XYZ<<<BlockPerGrid,ThreadPerBlock>>>(inputImg,outputImg,width,height);
}

__global__ void kRgb2XYZ(uchar4* inputImg, float4* outputImg, int width, int height)
{
	int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
	int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

    if(offset>=width*height)
        return;

	uchar4 nPixel=inputImg[offset];

	float _b=(float)nPixel.x/255.0f;
	float _g=(float)nPixel.y/255.0f;
	float _r=(float)nPixel.z/255.0f;

	float x=_r*0.412453f	+_g*0.357580f	+_b*0.180423f;
	float y=_r*0.212671f	+_g*0.715160f	+_b*0.072169f;
	float z=_r*0.019334f	+_g*0.119193f	+_b*0.950227f;

	float4 fPixel;
	fPixel.x=x;
	fPixel.y=y;
	fPixel.z=z;

	outputImg[offset]=fPixel;
}

