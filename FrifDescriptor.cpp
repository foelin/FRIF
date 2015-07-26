


#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "FrifDescriptor.h"
#include "utils.h"
#include "assert.h"
#include <algorithm>

using namespace cv;
using namespace std;

const unsigned int	FrifDescriptorExtractor::scales_		=	64;
const float			FrifDescriptorExtractor::scalerange_	=	30;     // 40->4 Octaves - else, this needs to be adjusted...
const unsigned int	FrifDescriptorExtractor::noRot_			=	1024;			// discretization of the rotation look-up
const unsigned int	FrifDescriptorExtractor::noPatternPoints_		=	60;		// number of pattern points
const unsigned int  FrifDescriptorExtractor::noTotalPairs_  = FrifDescriptorExtractor::noPatternPoints_*(FrifDescriptorExtractor::noPatternPoints_-1)/2;
const unsigned int  FrifDescriptorExtractor::noShortPairs_  = 512;
const unsigned int  FrifDescriptorExtractor::noLongPairs_	= FrifDescriptorExtractor::noTotalPairs_ - FrifDescriptorExtractor::noShortPairs_;
const unsigned int  FrifDescriptorExtractor::strings_		= 64;

//////////////////////////////////////////////////////////////////////////////

// constructors
FrifDescriptorExtractor::FrifDescriptorExtractor()
{
		generateKernel();
}

FrifDescriptorExtractor::FrifDescriptorExtractor(Params& params)
{
		params_ = params;
		generateKernel();
}


void FrifDescriptorExtractor::generateKernel()
{
	std::vector<float> radiusList;
	std::vector<int> numberList;

	// this is the standard pattern found to be suitable also
	const int ringNum = 5;
	float patternScale = params_.patchSize/FRIF_PATCH_SIZE;

	const float sigma_scale=1.3f;

	radiusList.resize(ringNum);
	numberList.resize(ringNum);
	const float f = 0.85f*patternScale;

	radiusList[0]=f*0.f;
	radiusList[1]=f*2.9f;
	radiusList[2]=f*4.9f;
	radiusList[3]=f*7.4f;
	radiusList[4]=f*10.8f;

	numberList[0]=1;
	numberList[1]=10;
	numberList[2]=14;
	numberList[3]=15;
	numberList[4]=20;

	assert(numberList[0]+numberList[1]+numberList[2]+numberList[3]+numberList[4]==noPatternPoints_);

	// set up the patterns
	patternPoints_=new FrifPatternPoint[noPatternPoints_*scales_*noRot_];
	FrifPatternPoint* patternIterator=patternPoints_;

	samplePos_ =new float[2*scales_*noRot_];

	// define the scale discretization:
	static const float lb_scale=log(scalerange_)/log(2.f);
	static const float lb_scale_step = lb_scale/(scales_);

	scaleList_=new float[scales_];
	sizeList_=new unsigned int[scales_];	//max range of pattern

	for(unsigned int scale = 0; scale <scales_; ++scale){
		scaleList_[scale]=pow(2.f,(float)(scale*lb_scale_step));
		sizeList_[scale]=0;

		// generate the pattern points look-up
		double alpha, theta;
		for(size_t rot=0; rot<noRot_; ++rot){
			theta = double(rot)*2*CV_PI/double(noRot_); // this is the rotation of the feature
			samplePos_[scale*(noRot_*2)+rot*2] = float(params_.lsRadius*scaleList_[scale]*cos(theta));
			samplePos_[scale*(noRot_*2)+rot*2+1] = float(params_.lsRadius*scaleList_[scale]*sin(theta));


			for(int ring = 0; ring<ringNum; ++ring){
				for(int num=0; num<numberList[ring]; ++num){
					// the actual coordinates on the circle
					alpha = (double(num))*2*CV_PI/double(numberList[ring]);
					patternIterator->x=float(scaleList_[scale]*radiusList[ring]*cos(alpha+theta)); // feature rotation plus angle of the point
					patternIterator->y=float(scaleList_[scale]*radiusList[ring]*sin(alpha+theta));
					// and the gaussian kernel sigma
					if(ring==0){
						patternIterator->sigma = sigma_scale*scaleList_[scale]*0.5f;
					}
					else
					{
						patternIterator->sigma = sigma_scale*scaleList_[scale]*(float(radiusList[ring]))*0.5f;
					}
					// adapt the sizeList if necessary
					const unsigned int size= (int)ceil(((scaleList_[scale]*(radiusList[ring]+params_.lsRadius))+patternIterator->sigma))+1;

					if(sizeList_[scale]<size){
						sizeList_[scale]=size;
					}

					// increment the iterator
					++patternIterator;
				}
			}
		}
	}

	frifPairs_.reserve(noTotalPairs_);

	for(unsigned int i= 1; i<noPatternPoints_; i++){
		for(unsigned int j= 0; j<i; j++){ //(find all the pairs)
			// point pair distance:
			const float dx=patternPoints_[j].x-patternPoints_[i].x;
			const float dy=patternPoints_[j].y-patternPoints_[i].y;
			FrifPair gbp;
			gbp.i = i;
			gbp.j = j;
			gbp.dist_sq = (dx*dx+dy*dy);
			gbp.weighted_dx = int((dx/(gbp.dist_sq))*2048.0+0.5);
			gbp.weighted_dy = int((dy/(gbp.dist_sq))*2048.0+0.5);
			frifPairs_.push_back(gbp);
		}
	}

	std::sort(frifPairs_.begin(), frifPairs_.end(), compareFrifPair);	//from long to small distance.
}




// simple alternative

__inline__ int FrifDescriptorExtractor::smoothedIntensity(const cv::Mat& image,
	const cv::Mat& integral,const float key_x,
	const float key_y, const unsigned int scale,
	const unsigned int rot, const unsigned int point) const{

		// get the float position
		const FrifPatternPoint& frifPoint = patternPoints_[scale*noRot_*noPatternPoints_ + rot*noPatternPoints_ + point];
		const float xf=frifPoint.x+key_x;
		const float yf=frifPoint.y+key_y;
		const int x = int(xf);
		const int y = int(yf);
		const int& imagecols=image.cols;

		// get the sigma:
		const float sigma_half=frifPoint.sigma;
		const float area=4.f*sigma_half*sigma_half;

		// calculate output:
		int ret_val;
		if(sigma_half<0.5){
			//interpolation multipliers:
			const int r_x=int((xf-x)*1024);
			const int r_y=int((yf-y)*1024);
			const int r_x_1=(1024-r_x);
			const int r_y_1=(1024-r_y);
			uchar* ptr=image.data+x+y*imagecols;
			// just interpolate:
			ret_val=(r_x_1*r_y_1*int(*ptr));
			ptr++;
			ret_val+=(r_x*r_y_1*int(*ptr));
			ptr+=imagecols;
			ret_val+=(r_x*r_y*int(*ptr));
			ptr--;
			ret_val+=(r_x_1*r_y*int(*ptr));
			return (ret_val+512)/1024;
		}

		// this is the standard case (simple, not speed optimized yet):

		// scaling:
		const int scaling = int(4194304.0/area);		//2^22 = 4194304
		const int scaling2= int(scaling*area/1024.f);//scaling2 = 2^12

		// the integral image is larger:
		const int integralcols=imagecols+1;

		// calculate borders
		const float x_1=xf-sigma_half;
		const float x1=xf+sigma_half;
		const float y_1=yf-sigma_half;
		const float y1=yf+sigma_half;

		const int x_left=int(x_1+0.5);
		const int y_top=int(y_1+0.5);
		const int x_right=int(x1+0.5);
		const int y_bottom=int(y1+0.5);

		// overlap area - multiplication factors:
		const float r_x_1=float(x_left)-x_1+0.5f;
		const float r_y_1=float(y_top)-y_1+0.5f;
		const float r_x1=x1-float(x_right)+0.5f;
		const float r_y1=y1-float(y_bottom)+0.5f;
		const int dx=x_right-x_left-1;
		const int dy=y_bottom-y_top-1;
		const int A=int((r_x_1*r_y_1)*scaling);
		const int B=int((r_x1*r_y_1)*scaling);
		const int C=int((r_x1*r_y1)*scaling);
		const int D=int((r_x_1*r_y1)*scaling);
		const int r_x_1_i=int(r_x_1*scaling);
		const int r_y_1_i=int(r_y_1*scaling);
		const int r_x1_i=int(r_x1*scaling);
		const int r_y1_i=int(r_y1*scaling);

		if(dx+dy>2){
			// now the calculation:
			uchar* ptr=image.data+x_left+imagecols*y_top;
			// first the corners:
			ret_val=A*int(*ptr);
			ptr+=dx+1;
			ret_val+=B*int(*ptr);
			ptr+=dy*imagecols+1;
			ret_val+=C*int(*ptr);
			ptr-=dx+1;
			ret_val+=D*int(*ptr);

			// next the edges:
			int* ptr_integral=(int*)integral.data+x_left+integralcols*y_top+1;
			// find a simple path through the different surface corners

			/*
			    ____
			 __|	|__
			|          |
			|__  	 __|
			   |____|
			*/
			const int tmp1=(*ptr_integral);
			ptr_integral+=dx;
			const int tmp2=(*ptr_integral);
			ptr_integral+=integralcols;
			const int tmp3=(*ptr_integral);
			ptr_integral++;
			const int tmp4=(*ptr_integral);
			ptr_integral+=dy*integralcols;
			const int tmp5=(*ptr_integral);
			ptr_integral--;
			const int tmp6=(*ptr_integral);
			ptr_integral+=integralcols;
			const int tmp7=(*ptr_integral);
			ptr_integral-=dx;
			const int tmp8=(*ptr_integral);
			ptr_integral-=integralcols;
			const int tmp9=(*ptr_integral);
			ptr_integral--;
			const int tmp10=(*ptr_integral);
			ptr_integral-=dy*integralcols;
			const int tmp11=(*ptr_integral);
			ptr_integral++;
			const int tmp12=(*ptr_integral);

			// assign the weighted surface integrals:
			const int upper=(tmp3-tmp2+tmp1-tmp12)*r_y_1_i;
			const int middle=(tmp6-tmp3+tmp12-tmp9)*scaling;
			const int left=(tmp9-tmp12+tmp11-tmp10)*r_x_1_i;
			const int right=(tmp5-tmp4+tmp3-tmp6)*r_x1_i;
			const int bottom=(tmp7-tmp6+tmp9-tmp8)*r_y1_i;

			return (ret_val+upper+middle+left+right+bottom+scaling2/2)/scaling2;
		}

		// now the calculation:
		uchar* ptr=image.data+x_left+imagecols*y_top;
		// first row:
		ret_val=A*int(*ptr);
		ptr++;
		const uchar* end1 = ptr+dx;
		for(; ptr<end1; ptr++){
			ret_val+=r_y_1_i*int(*ptr);
		}
		ret_val+=B*int(*ptr);
		// middle ones:
		ptr+=imagecols-dx-1;
		uchar* end_j=ptr+dy*imagecols;
		for(; ptr<end_j; ptr+=imagecols-dx-1){
			ret_val+=r_x_1_i*int(*ptr);
			ptr++;
			const uchar* end2 = ptr+dx;
			for(; ptr<end2; ptr++){
				ret_val+=int(*ptr)*scaling;
			}
			ret_val+=r_x1_i*int(*ptr);
		}
		// last row:
		ret_val+=D*int(*ptr);
		ptr++;
		const uchar* end3 = ptr+dx;
		for(; ptr<end3; ptr++){
			ret_val+=r_y1_i*int(*ptr);
		}
		ret_val+=C*int(*ptr);

		return (ret_val+scaling2/2)/scaling2;
}

// simple alternative:
__inline__ int FrifDescriptorExtractor::smoothedIntensity(const cv::Mat& image,
	const cv::Mat& integral,const float key_x,
	const float key_y, const unsigned int scale,
	const unsigned int rot, const unsigned int point, const float delta_x, const float delta_y) const{

		// get the float position
		const FrifPatternPoint& frifPoint = patternPoints_[scale*noRot_*noPatternPoints_ + rot*noPatternPoints_ + point];
		const float xf=frifPoint.x+key_x+delta_x;
		const float yf=frifPoint.y+key_y+delta_y;
		const int x = int(xf);
		const int y = int(yf);
		const int& imagecols=image.cols;

		// get the sigma:
		const float sigma_half=frifPoint.sigma;
		const float area=4.f*sigma_half*sigma_half;

		// calculate output:
		int ret_val;
		if(sigma_half<0.5){
			//interpolation multipliers:
			const int r_x=int((xf-x)*1024);
			const int r_y=int((yf-y)*1024);
			const int r_x_1=(1024-r_x);
			const int r_y_1=(1024-r_y);
			uchar* ptr=image.data+x+y*imagecols;
			// just interpolate:
			ret_val=(r_x_1*r_y_1*int(*ptr));
			ptr++;
			ret_val+=(r_x*r_y_1*int(*ptr));
			ptr+=imagecols;
			ret_val+=(r_x*r_y*int(*ptr));
			ptr--;
			ret_val+=(r_x_1*r_y*int(*ptr));
			return (ret_val+512)/1024;
		}

		// this is the standard case (simple, not speed optimized yet):

		// scaling:
		const int scaling = int(4194304.0/area);
		const int scaling2=int(scaling*area/1024.0);

		// the integral image is larger:
		const int integralcols=imagecols+1;

		// calculate borders
		const float x_1=xf-sigma_half;
		const float x1=xf+sigma_half;
		const float y_1=yf-sigma_half;
		const float y1=yf+sigma_half;

		const int x_left=int(x_1+0.5);
		const int y_top=int(y_1+0.5);
		const int x_right=int(x1+0.5);
		const int y_bottom=int(y1+0.5);

		// overlap area - multiplication factors:
		const float r_x_1=float(x_left)-x_1+0.5f;
		const float r_y_1=float(y_top)-y_1+0.5f;
		const float r_x1=x1-float(x_right)+0.5f;
		const float r_y1=y1-float(y_bottom)+0.5f;
		const int dx=x_right-x_left-1;
		const int dy=y_bottom-y_top-1;
		const int A=int((r_x_1*r_y_1)*scaling);
		const int B=int((r_x1*r_y_1)*scaling);
		const int C=int((r_x1*r_y1)*scaling);
		const int D=int((r_x_1*r_y1)*scaling);
		const int r_x_1_i=int(r_x_1*scaling);
		const int r_y_1_i=int(r_y_1*scaling);
		const int r_x1_i=int(r_x1*scaling);
		const int r_y1_i=int(r_y1*scaling);

		if(dx+dy>2){
			// now the calculation:
			uchar* ptr=image.data+x_left+imagecols*y_top;
			// first the corners:
			ret_val=A*int(*ptr);
			ptr+=dx+1;
			ret_val+=B*int(*ptr);
			ptr+=dy*imagecols+1;
			ret_val+=C*int(*ptr);
			ptr-=dx+1;
			ret_val+=D*int(*ptr);

			// next the edges:
			int* ptr_integral=(int*)integral.data+x_left+integralcols*y_top+1;
			// find a simple path through the different surface corners
			const int tmp1=(*ptr_integral);
			ptr_integral+=dx;
			const int tmp2=(*ptr_integral);
			ptr_integral+=integralcols;
			const int tmp3=(*ptr_integral);
			ptr_integral++;
			const int tmp4=(*ptr_integral);
			ptr_integral+=dy*integralcols;
			const int tmp5=(*ptr_integral);
			ptr_integral--;
			const int tmp6=(*ptr_integral);
			ptr_integral+=integralcols;
			const int tmp7=(*ptr_integral);
			ptr_integral-=dx;
			const int tmp8=(*ptr_integral);
			ptr_integral-=integralcols;
			const int tmp9=(*ptr_integral);
			ptr_integral--;
			const int tmp10=(*ptr_integral);
			ptr_integral-=dy*integralcols;
			const int tmp11=(*ptr_integral);
			ptr_integral++;
			const int tmp12=(*ptr_integral);

			// assign the weighted surface integrals:
			const int upper=(tmp3-tmp2+tmp1-tmp12)*r_y_1_i;
			const int middle=(tmp6-tmp3+tmp12-tmp9)*scaling;
			const int left=(tmp9-tmp12+tmp11-tmp10)*r_x_1_i;
			const int right=(tmp5-tmp4+tmp3-tmp6)*r_x1_i;
			const int bottom=(tmp7-tmp6+tmp9-tmp8)*r_y1_i;

			return (ret_val+upper+middle+left+right+bottom+scaling2/2)/scaling2;
		}

		// now the calculation:
		uchar* ptr=image.data+x_left+imagecols*y_top;
		// first row:
		ret_val=A*int(*ptr);
		ptr++;
		const uchar* end1 = ptr+dx;
		for(; ptr<end1; ptr++){
			ret_val+=r_y_1_i*int(*ptr);
		}
		ret_val+=B*int(*ptr);
		// middle ones:
		ptr+=imagecols-dx-1;
		uchar* end_j=ptr+dy*imagecols;
		for(; ptr<end_j; ptr+=imagecols-dx-1){
			ret_val+=r_x_1_i*int(*ptr);
			ptr++;
			const uchar* end2 = ptr+dx;
			for(; ptr<end2; ptr++){
				ret_val+=int(*ptr)*scaling;
			}
			ret_val+=r_x1_i*int(*ptr);
		}
		// last row:
		ret_val+=D*int(*ptr);
		ptr++;
		const uchar* end3 = ptr+dx;
		for(; ptr<end3; ptr++){
			ret_val+=r_y1_i*int(*ptr);
		}
		ret_val+=C*int(*ptr);

		return (ret_val+scaling2/2)/scaling2;
}


// simply take average on a square patch, not even gaussian approx
__inline__ int FrifDescriptorExtractor::smoothedIntensity_fast( const cv::Mat& image, const cv::Mat& integral,
	const float kp_x,
	const float kp_y,
	const unsigned int scale,
	const unsigned int rot,
	const unsigned int point) const
{
		// get point position in image
		const FrifPatternPoint& frifPoint = patternPoints_[scale*noRot_*noPatternPoints_ + rot*noPatternPoints_ + point];
		const float xf = frifPoint.x+kp_x;
		const float yf = frifPoint.y+kp_y;
		const int x = int(xf);
		const int y = int(yf);
		const int& imagecols = image.cols;

		// get the sigma:
		const float radius = frifPoint.sigma;

		// calculate output:
		int ret_val;
		if( radius < 0.5 ) {
			// interpolation multipliers:
			const int r_x = int((xf-x)*1024);
			const int r_y = int((yf-y)*1024);
			const int r_x_1 = int((1024-r_x));
			const int r_y_1 = int((1024-r_y));
			uchar* ptr = image.data+x+y*imagecols;
			// linear interpolation:
			ret_val = (r_x_1*r_y_1*int(*ptr));
			ptr++;
			ret_val += (r_x*r_y_1*int(*ptr));
			ptr += imagecols;
			ret_val += (r_x*r_y*int(*ptr));
			ptr--;
			ret_val += (r_x_1*r_y*int(*ptr));
			return (ret_val+512)/1024;
		}

		// expected case:

		// calculate borders
		const int x_left = int(xf-radius+0.5);
		const int y_top = int(yf-radius+0.5);
		const int x_right = int(xf+radius+1.5);//integral image is 1px wider
		const int y_bottom = int(yf+radius+1.5);//integral image is 1px higher


		ret_val = integral.at<int>(y_bottom,x_right);//bottom right corner
		ret_val -= integral.at<int>(y_bottom,x_left);
		ret_val += integral.at<int>(y_top,x_left);
		ret_val -= integral.at<int>(y_top,x_right);
		ret_val = ret_val/( (x_right-x_left)* (y_bottom-y_top) );

		//~ std::cout<<integral.step[1]<<std::endl;
		return ret_val;
}

__inline__ int FrifDescriptorExtractor::smoothedIntensity_fast( const cv::Mat& image, const cv::Mat& integral,
	const float kp_x,
	const float kp_y,
	const unsigned int scale,
	const unsigned int rot,
	const unsigned int point,
	const float deltaX,
	const float deltaY) const
{
		// get point position in image
		const FrifPatternPoint& frifPoint = patternPoints_[scale*noRot_*noPatternPoints_ + rot*noPatternPoints_ + point];
		const float xf = frifPoint.x+kp_x+deltaX;
		const float yf = frifPoint.y+kp_y+deltaY;
		const int x = int(xf);
		const int y = int(yf);
		const int& imagecols = image.cols;

		// get the sigma:
		const float radius = frifPoint.sigma;

		// calculate output:
		int ret_val;
		if( radius < 0.5 ) {
			// interpolation multipliers:
			const int r_x = int((xf-x)*1024);
			const int r_y = int((yf-y)*1024);
			const int r_x_1 = (1024-r_x);
			const int r_y_1 = (1024-r_y);
			uchar* ptr = image.data+x+y*imagecols;
			// linear interpolation:
			ret_val = (r_x_1*r_y_1*int(*ptr));
			ptr++;
			ret_val += (r_x*r_y_1*int(*ptr));
			ptr += imagecols;
			ret_val += (r_x*r_y*int(*ptr));
			ptr--;
			ret_val += (r_x_1*r_y*int(*ptr));
			return (ret_val+512)/1024;
		}

		// expected case:

		// calculate borders
		const int x_left = int(xf-radius+0.5);
		const int y_top = int(yf-radius+0.5);
		const int x_right = int(xf+radius+1.5);//integral image is 1px wider
		const int y_bottom = int(yf+radius+1.5);//integral image is 1px higher


		ret_val = integral.at<int>(y_bottom,x_right);//bottom right corner
		ret_val -= integral.at<int>(y_bottom,x_left);
		ret_val += integral.at<int>(y_top,x_left);
		ret_val -= integral.at<int>(y_top,x_right);

		int* integral_data = (int*)integral.data;
		int step = integral.step/sizeof(int);
		ret_val = integral_data[y_bottom*step+x_right];
		ret_val -= integral_data[y_bottom*step+x_left];
		ret_val += integral_data[y_top*step+x_left];
		ret_val -= integral_data[y_top*step+x_right];

			ret_val = ret_val/( (x_right-x_left)* (y_bottom-y_top) );

		//~ std::cout<<integral.step[1]<<std::endl;
		return ret_val;
}

// computes the descriptor
void FrifDescriptorExtractor::computeImpl(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const
{

	// first, calculate the integral image over the whole image:
	// current integral image
	cv::Mat _integral; // the integral image
	cv::integral(image, _integral);

	int* ptr_integral=(int*)_integral.data;

	std::vector<int> kscales; // remember the scale per keypoint

	//Remove keypoints very close to the border
	size_t ksize=keypoints.size();

	kscales.resize(ksize);
	static const float log2 = 0.693147180559945f;
	static const float lb_scalerange = log(scalerange_)/(log2);
	static const float lb_scale_step = lb_scalerange/(scales_);


	std::vector<KeyPoint>::iterator beginning = keypoints.begin();
	std::vector<int>::iterator beginningkscales = kscales.begin();
	unsigned int basicscale=0;

	for(size_t k=0; k<ksize; k++){
		unsigned int scale;
		scale=max((int)(1.f/lb_scale_step*(log(keypoints[k].size/(FRIF_BASE_SIZE))/log2)+0.5),0);
		// saturate
		if(scale>=scales_) scale = scales_-1;
		kscales[k]=scale;

		const int border = sizeList_[scale];
		const int border_x=image.cols-border;
		const int border_y=image.rows-border;
		if(isOutOfBound((float)border, (float)border, (float)border_x, (float)border_y, keypoints[k])){
			keypoints[k].octave = -1;
		}
	}

	size_t count=0;
	for (size_t k=0; k<ksize; k++)
	{
		KeyPoint& src = keypoints[k];
		if (src.octave == -1)
			continue;

		if (k!=count)
		{
			keypoints[count] = src;
			kscales[count] = kscales[k];
		}
		count++;
	}
	keypoints.resize(count);
	kscales.resize(count);


	// the feature orientation
	computeDomiOrien(image, _integral, keypoints, kscales);

	//compute descriptor
	computeFrif(image, _integral, keypoints, kscales, descriptors);

	// clean-up
	_integral.release();

}

int FrifDescriptorExtractor::descriptorSize() const{
	return strings_;
}

int FrifDescriptorExtractor::descriptorType() const{
	return CV_8U;
}

FrifDescriptorExtractor::~FrifDescriptorExtractor(){
	delete [] patternPoints_;
	delete [] scaleList_;
	delete [] sizeList_;
}

void FrifDescriptorExtractor::computeDomiOrien(const Mat& image, const Mat& integral,
	std::vector<KeyPoint>& keypoints, std::vector<int>& kscales) const
{
	int t1;
	int t2;
	int direction0;
	int direction1;

	size_t ksize=keypoints.size();
	int _values[noPatternPoints_]; // for temporary use

	for(size_t k=0; k<ksize; k++)
	{
		KeyPoint& kp=keypoints[k];
		const int& scale=kscales[k];
		int shifter=0;
		int* pvalues =_values;
		const float& x=kp.pt.x;
		const float& y=kp.pt.y;

		// get the gray values in the unrotated pattern
		for(unsigned int i = 0; i<noPatternPoints_; i++)
		{
			*(pvalues++)=smoothedIntensity_fast(image, integral, x,
				y, scale, 0, i);
		}

		direction0=0;
		direction1=0;
		// now iterate through the long pairings
		for(int i=0; i<noLongPairs_; i++)
		{
			const FrifPair& cur = frifPairs_[i];
			t1=*(_values+cur.i);
			t2=*(_values+cur.j);
			const int delta_t=(t1-t2);
			// update the direction:
			const int tmp0=delta_t*(cur.weighted_dx)/1024;
			const int tmp1=delta_t*(cur.weighted_dy)/1024;
			direction0+=tmp0;
			direction1+=tmp1;
		}
		kp.angle = float(atan2((float)direction1,(float)direction0)/CV_PI*180.0);
	}
}

void FrifDescriptorExtractor::computeFrif(const Mat& image, const Mat& integral,std::vector<KeyPoint>& keypoints,
	std::vector<int>& kscales, Mat& descriptors) const
{
	int t1,t2;
	size_t ksize=keypoints.size();

	// resize the descriptors:
	descriptors=cv::Mat::zeros(ksize,strings_, CV_8U);

	int _values[5*noPatternPoints_]; // for temporary use

	uchar* ptr = descriptors.data;
	for(size_t k=0; k<ksize; k++)
	{
		int theta;
		KeyPoint& kp=keypoints[k];
		const int& scale=kscales[k];

		int shifter=0;
		int* pvalues =_values;
		const float& x=kp.pt.x;
		const float& y=kp.pt.y;


		theta=int((noRot_*kp.angle)/(360.0)+0.5);
		if(theta<0)
			theta+=noRot_;
		if(theta>=int(noRot_))
			theta-=noRot_;

		// now also extract the stuff for the actual direction:
		// let us compute the smoothed values
		shifter=0;

		//unsigned int mean=0;
		pvalues =_values;
		// get the gray values in the rotated pattern
		float deltaX=0.f;
		float deltaY=0.f;

		//rotation invariant
		float* offset = &samplePos_[scale*(noRot_*2)];
		for(unsigned int i = 0; i<noPatternPoints_; i++){
			for (unsigned j=0; j<4; j++)
			{
				int sampleOri = cvRound((noRot_*(kp.angle+90*j))/360.0);
				if(sampleOri<0)
					sampleOri+=noRot_;
				if(sampleOri>=int(noRot_))
					sampleOri-=noRot_;

				deltaX = offset[sampleOri*2];
				deltaY = offset[sampleOri*2+1];

				*(pvalues++)=smoothedIntensity_fast(image, integral, x, y, scale, theta, i, deltaX, deltaY);
			}
			*(pvalues++)=smoothedIntensity_fast(image, integral, x, y, scale, theta, i);
		}

		// now iterate through all the pairings
		uchar* ptr2= ptr;
		for (unsigned int i=0; i<noPatternPoints_; i++)
		{
			for (int s=0; s<3; s++)
			{
				for (int t=3;t>s; t--)
				{
					if(_values[5*i+s] > _values[5*i+t])
						*ptr2 |= ((1)<<shifter);
					shifter++;

					if(shifter==8){
						shifter=0;
						++ptr2;
					}
				}
			}
		}

		int num = 512 - noPatternPoints_*6;
		int minIdx = noTotalPairs_ - num;
		for(int i=noTotalPairs_-1; i>=minIdx; i--)
		{
			const FrifPair& cur = frifPairs_[i];
			t1=_values[5*cur.i+4];
			t2=_values[5*cur.j+4];
			if(t1>t2){
				*ptr2|=((1)<<shifter);

			}
			++shifter;
			if(shifter==8){
				shifter=0;
				++ptr2;
			}
		}
		ptr+=strings_;

	}
}
