#include "FrifDetector.h"
#include "utils.h"
#include "assert.h"

using namespace cv;

const float FrifScaleSpace::safetyFactor_          =1.f;
const float FrifScaleSpace::basicSize_             =12.f;



/*********************************************************************************************

	GeneralFeatureDetector

**********************************************************************************************/

FrifFeatureDetector::FrifFeatureDetector(Params& params){
	this->params_ = params;
}

void FrifFeatureDetector::detectImpl( const cv::Mat& image,
	std::vector<cv::KeyPoint>& keypoints,
	const cv::Mat& mask) const
{
	FrifScaleSpace frifScaleSpace(params_);
	frifScaleSpace.constructPyramid(image);
	frifScaleSpace.getKeypoints(keypoints);

	// remove invalid points
	//removeInvalidPoints(mask, keypoints);
}

void FrifFeatureDetector::setThresh(float thresh)
{
	params_.blobThresh = thresh;
}


void FrifFeatureDetector::detect( const Mat&  image,
                                  std::vector<KeyPoint>& keypoints,
                                 const Mat&  mask)

    {
        detectImpl(image,keypoints,mask );
    }

/*********************************************************************************************

	FrifScaleSpace

**********************************************************************************************/
// construct telling the octaves number:
FrifScaleSpace::FrifScaleSpace(const Params& params){
	if(params.octaves==0)
		layers_=1;
	else
		layers_=2*params.octaves;

	this->params_ = params;
}
FrifScaleSpace::~FrifScaleSpace(){

	pyramid_.clear();
}
// construct the image pyramids
void FrifScaleSpace::constructPyramid(const cv::Mat& image){

	// set correct size:
	pyramid_.clear();

	// fill the pyramid:
	pyramid_.push_back(FrifLayer(image.clone(), params_));
	if(layers_>1){
		pyramid_.push_back(FrifLayer(pyramid_.back(),FrifLayer::CommonParams::TWOTHIRDSAMPLE, params_));
	}
	const int octaves2=layers_;

	for(uint8_t i=2; i<octaves2; i+=2){
		pyramid_.push_back(FrifLayer(pyramid_[i-2],FrifLayer::CommonParams::HALFSAMPLE, params_));
		pyramid_.push_back(FrifLayer(pyramid_[i-1],FrifLayer::CommonParams::HALFSAMPLE, params_));
	}
}

void FrifScaleSpace::getKeypoints(std::vector<cv::KeyPoint>& keypoints)
{

		if (layers_ == 1)
			getBlob(keypoints);
		else
			getBlobBlob(keypoints);

}

__inline__ bool FrifScaleSpace::isMax2D(const int x_layer, const int y_layer, const cv::Mat& scores){
	const int scorescols = scores.cols;
	float* scores_data = (float*)scores.data;
	float* data=scores_data + y_layer*scorescols + x_layer;

	// decision tree:
	const float center = (*data);
	data--;
	const float s_10=*data;
	if(center<=s_10) return false;
	data+=2;
	const float s10=*data;
	if(center<=s10) return false;
	data-=(scorescols+1);
	const float s0_1=*data;
	if(center<=s0_1) return false;
	data+=2*scorescols;
	const float s01=*data;
	if(center<=s01) return false;
	data--;
	const float s_11=*data;
	if(center<=s_11) return false;
	data+=2;
	const float s11=*data;
	if(center<=s11) return false;
	data-=2*scorescols;
	const float s1_1=*data;
	if(center<=s1_1) return false;
	data-=2;
	const float s_1_1=*data;
	if(center<=s_1_1) return false;

	return true;
}

// 3D maximum refinement centered around (x_layer,y_layer)
__inline__ float FrifScaleSpace::refine3DBlobBlob(const uint8_t layer,
	const int x_layer, const int y_layer,
	float& x, float& y, float& scale,  bool& ismax)
{
	ismax = false;

	FrifLayer& thisLayer=pyramid_[layer];
	const float center = thisLayer.getBlobScore(x_layer,y_layer);

	float x_center, y_center, blob_center, x_above, y_above, blob_above, x_below, y_below, blob_below;
	float blob;
	float scale_delta;

	// get the patch on this layer:
	register float s_0_0 = thisLayer.getBlobScore(x_layer-1, y_layer-1);
	register float s_1_0 = thisLayer.getBlobScore(x_layer,   y_layer-1);
	register float s_2_0 = thisLayer.getBlobScore(x_layer+1, y_layer-1);
	register float s_2_1 = thisLayer.getBlobScore(x_layer+1, y_layer);
	register float s_1_1 = thisLayer.getBlobScore(x_layer,   y_layer);
	register float s_0_1 = thisLayer.getBlobScore(x_layer-1, y_layer);
	register float s_0_2 = thisLayer.getBlobScore(x_layer-1, y_layer+1);
	register float s_1_2 = thisLayer.getBlobScore(x_layer,   y_layer+1);
	register float s_2_2 = thisLayer.getBlobScore(x_layer+1, y_layer+1);

	float delta_x_layer, delta_y_layer;

	blob_center = subpixel2DMax(s_0_0, s_0_1, s_0_2,s_1_0, s_1_1, s_1_2,s_2_0, s_2_1, s_2_2,
		delta_x_layer, delta_y_layer);

	x_center = (float)x_layer+delta_x_layer;
	y_center = (float)y_layer+delta_y_layer;

	if (layer%2 == 0)//octave
	{

		x_above = (x_center+1)*2.f/3.f-1;
		y_above = (y_center+1)*2.f/3.f-1;

		x_below = (x_center+1)*4.f/3.f-1;
		y_below = (y_center+1)*4.f/3.f-1;
	}
	else
	{
		x_above = (x_center+1)*3.f/4.f-1;
		y_above = (y_center+1)*3.f/4.f-1;

		x_below = (x_center+1)*3.f/2.f-1;
		y_below = (y_center+1)*3.f/2.f-1;
	}

	if (layer == layers_-1)
	{
		FrifLayer& belowLayer = pyramid_[layer-1];
		blob_below = belowLayer.getBlobScore(x_below,y_below);

		if (blob_below <= blob_center)
			ismax = true;
		else
		{
			ismax = false;
			return 0.f;
		}
		blob = blob_center;
		scale_delta = 1.f;

	}
	else if (layer == 0)
	{

		FrifLayer& aboveLayer = pyramid_[layer+1];
		blob_above = aboveLayer.getBlobScore(x_above, y_above);

		blob_below = thisLayer.computeBlobScore_5_8(x_center, y_center);

		if (blob_center>=blob_above&&blob_center>=blob_below)
			ismax = true;
		else
		{
			ismax = false;
			return 0.f;
		}

		scale_delta=refine1D_2(blob_below, blob_center, blob_above, blob);
	}
	else
	{
		FrifLayer& aboveLayer = pyramid_[layer+1];
		blob_above = aboveLayer.getBlobScore(x_above, y_above);

		FrifLayer& belowLayer = pyramid_[layer-1];
		blob_below = belowLayer.getBlobScore(x_below,y_below);

		if (blob_center>=blob_above&&blob_center>=blob_below)
			ismax = true;
		else
		{
			ismax = false;
			return 0.f;
		}
		if (layer%2 == 0)//octave
			scale_delta=refine1D(blob_below, blob_center, blob_above, blob);
		else	//intra octave
			scale_delta=refine1D_1(blob_below,blob_center,blob_above,blob);
	}

	x = x_center*thisLayer.scale()+thisLayer.offset();
	y = y_center*thisLayer.scale()+thisLayer.offset();
	scale = scale_delta*thisLayer.scale();

	// that's it, return the refined maximum:
	return blob;

}

/*
fit y=ax^2+bx+c
input points:(0.75,s_05), (1.0, s0), (1.5, s05) [currently in the octave i(i!=0)]

3a =  16*s_05 - 24*s0 + 8*s05
3b = -40*s_05 + 54*s0 - 14*s05
3c =  24*s_05 - 27*s0 + 6*s05

*/
__inline__ float FrifScaleSpace::refine1D(const float s_05,
	const float s0, const float s05, float& max){
		int i_05=int(1024.f*s_05+0.5f);
		int i0=int(1024.f*s0+0.5f);
		int i05=int(1024.f*s05+0.5f);

		//   16.0000  -24.0000    8.0000
		//  -40.0000   54.0000  -14.0000
		//   24.0000  -27.0000    6.0000

		int three_a=16*i_05-24*i0+8*i05;
		// second derivative must be negative:

		if(three_a>=0){
			if(s0>=s_05 && s0>=s05){
				max=s0;
				return 1.f;
			}
			if(s_05>=s0 && s_05>=s05){
				max=s_05;
				return 0.75;
			}
			if(s05>=s0 && s05>=s_05){
				max=s05;
				return 1.5f;
			}
		}

		int three_b=-40*i_05+54*i0-14*i05;
		// calculate max location:
		float ret_val=-float(three_b)/float(2*three_a);
		// saturate and return
		if(ret_val<0.75) ret_val= 0.75;
		else if(ret_val>1.5f) ret_val= 1.5f; // allow to be slightly off bounds ...?
		int three_c = +24*i_05  -27*i0    +6*i05;
		max=float(three_c)+float(three_a)*ret_val*ret_val+float(three_b)*ret_val;
		max/=3072.f;
		return ret_val;
}

/*
fit y=ax^2+bx+c
input points are (2/3,s_05), (1.0, s0), (4/3, s05) [currently in the intra octave]
*/
__inline__ float FrifScaleSpace::refine1D_1(const float s_05,
	const float s0, const float s05, float& max){
		int i_05=int(1024.f*s_05+0.5f);
		int i0=int(1024.f*s0+0.5f);
		int i05=int(1024.f*s05+0.5f);

		//  4.5000   -9.0000    4.5000
		//-10.5000   18.0000   -7.5000
		//  6.0000   -8.0000    3.0000

		int two_a=9*i_05-18*i0+9*i05;
		// second derivative must be negative:
		if(two_a>=0){
			if(s0>=s_05 && s0>=s05){
				max=s0;
				return 1.f;
			}
			if(s_05>=s0 && s_05>=s05){
				max=s_05;
				return 0.6666666666666666666666666667f;
			}
			if(s05>=s0 && s05>=s_05){
				max=s05;
				return 1.3333333333333333333333333333f;
			}
		}

		int two_b=-21*i_05+36*i0-15*i05;
		// calculate max location:
		float ret_val=-float(two_b)/float(2*two_a);
		// saturate and return
		if(ret_val<0.6666666666666666666666666667f) ret_val= 0.666666666666666666666666667f;
		else if(ret_val>1.33333333333333333333333333f) ret_val= 1.333333333333333333333333333f;
		int two_c = +12*i_05  -16*i0    +6*i05;
		max=float(two_c)+float(two_a)*ret_val*ret_val+float(two_b)*ret_val;
		max/=2048.f;
		return ret_val;
}

/*
fit y=ax^2+bx+c
input points are (0.7,s_05), (1.0, s0), (1.5, s05) [currently in the octave 0]
*/
__inline__ float FrifScaleSpace::refine1D_2(const float s_05,
	const float s0, const float s05, float& max){
		int i_05=int(1024.f*s_05+0.5f);
		int i0=int(1024.f*s0+0.5f);
		int i05=int(1024.f*s05+0.5f);

		//   18.0000  -30.0000   12.0000
		//  -45.0000   65.0000  -20.0000
		//   27.0000  -30.0000    8.0000

		int a=2*i_05-4*i0+2*i05;
		// second derivative must be negative:
		if(a>=0){
			if(s0>=s_05 && s0>=s05){
				max=s0;
				return 1.f;
			}
			if(s_05>=s0 && s_05>=s05){
				max=s_05;
				return 0.7f;
			}
			if(s05>=s0 && s05>=s_05){
				max=s05;
				return 1.5f;
			}
		}

		int b=-5*i_05+8*i0-3*i05;
		// calculate max location:
		float ret_val=-float(b)/float(2*a);
		// saturate and return
		if(ret_val<0.7) ret_val= 0.7f;
		else if(ret_val>1.5f) ret_val= 1.5f; // allow to be slightly off bounds ...?
		int c = +3*i_05  -3*i0    +1*i05;
		max=float(c)+float(a)*ret_val*ret_val+float(b)*ret_val;
		max/=1024;
		return ret_val;
}

/*
fit z=a1*x^2+a2*y^2+a3*x+a4*y+a5*xy+a6
*/
__inline__ float FrifScaleSpace::subpixel2DMax(const int s_0_0, const int s_0_1, const int s_0_2,
	const int s_1_0, const int s_1_1, const int s_1_2,
	const int s_2_0, const int s_2_1, const int s_2_2,
	float& delta_x, float& delta_y){

		// the coefficients of the 2d quadratic function least-squares fit:
		register int tmp1 =        s_0_0 + s_0_2 - 2*s_1_1 + s_2_0 + s_2_2;
		register int coeff1 = 3*(tmp1 + s_0_1 - ((s_1_0 + s_1_2)<<1) + s_2_1);
		register int coeff2 = 3*(tmp1 - ((s_0_1+ s_2_1)<<1) + s_1_0 + s_1_2 );
		register int tmp2 =                                  s_0_2 - s_2_0;
		register int tmp3 =                         (s_0_0 + tmp2 - s_2_2);
		register int tmp4 =                                   tmp3 -2*tmp2;
		register int coeff3 =                    -3*(tmp3 + s_0_1 - s_2_1);
		register int coeff4 =                    -3*(tmp4 + s_1_0 - s_1_2);
		register int coeff5 =            (s_0_0 - s_0_2 - s_2_0 + s_2_2)<<2;
		register int coeff6 = -(s_0_0  + s_0_2 - ((s_1_0 + s_0_1 + s_1_2 + s_2_1)<<1) - 5*s_1_1  + s_2_0  + s_2_2)<<1;


		// 2nd derivative test:
		register int H_det=4*coeff1*coeff2 - coeff5*coeff5;

		if(H_det==0){
			delta_x=0.f;
			delta_y=0.f;
			return float(coeff6)/18.f;
		}

		if(!(H_det>0&&coeff1<0)){
			// The maximum must be at the one of the 4 patch corners.
			int tmp_max=coeff3+coeff4+coeff5;
			delta_x=1.f; delta_y=1.f;

			int tmp = -coeff3+coeff4-coeff5;
			if(tmp>tmp_max){
				tmp_max=tmp;
				delta_x=-1.f; delta_y=1.f;
			}
			tmp = coeff3-coeff4-coeff5;
			if(tmp>tmp_max){
				tmp_max=tmp;
				delta_x=1.f; delta_y=-1.f;
			}
			tmp = -coeff3-coeff4+coeff5;
			if(tmp>tmp_max){
				tmp_max=tmp;
				delta_x=-1.f; delta_y=-1.f;
			}
			return float(tmp_max+coeff1+coeff2+coeff6)/18.f;
		}

		// this is hopefully the normal outcome of the Hessian test

		delta_x=float(2*coeff2*coeff3 - coeff4*coeff5)/float(-H_det);
		delta_y=float(2*coeff1*coeff4 - coeff3*coeff5)/float(-H_det);


		// TODO: this is not correct, but easy, so perform a real boundary maximum search:
		bool tx=false; bool tx_=false; bool ty=false; bool ty_=false;
		if(delta_x>1.f) tx=true;
		else if(delta_x<-1.f) tx_=true;
		if(delta_y>1.f) ty=true;
		else if(delta_y<-1.f) ty_=true;

		if(tx||tx_||ty||ty_){	//the extrema is not in [-1,1], then search for it
			// get two candidates:
			float delta_x1=0.f, delta_x2=0.f, delta_y1=0.f, delta_y2=0.f;
			if(tx) {
				delta_x1=1.f;
				delta_y1=-float(coeff4+coeff5)/float(2*coeff2);
				if(delta_y1>1.f) delta_y1=1.f; else if (delta_y1<-1.f) delta_y1=-1.f;
			}
			else if(tx_) {
				delta_x1=-1.f;
				delta_y1=-float(coeff4-coeff5)/float(2*coeff2);
				if(delta_y1>1.f) delta_y1=1.f; else if (delta_y1<-1.f) delta_y1=-1.f;
			}
			if(ty) {
				delta_y2=1.f;
				delta_x2=-float(coeff3+coeff5)/float(2*coeff1);
				if(delta_x2>1.f) delta_x2=1.f; else if (delta_x2<-1.f) delta_x2=-1.f;
			}
			else if(ty_) {
				delta_y2=-1.f;
				delta_x2=-float(coeff3-coeff5)/float(2*coeff1);
				if(delta_x2>1.f) delta_x2=1.f; else if (delta_x2<-1.f) delta_x2=-1.f;
			}
			// insert both options for evaluation which to pick
			float max1 = (coeff1*delta_x1*delta_x1+coeff2*delta_y1*delta_y1
				+coeff3*delta_x1+coeff4*delta_y1
				+coeff5*delta_x1*delta_y1
				+coeff6)/18.f;
			float max2 = (coeff1*delta_x2*delta_x2+coeff2*delta_y2*delta_y2
				+coeff3*delta_x2+coeff4*delta_y2
				+coeff5*delta_x2*delta_y2
				+coeff6)/18.f;


			if(max1>max2) {
				delta_x=delta_x1;
				delta_y=delta_y1;
				return max1;
			}
			else{
				delta_x=delta_x2;
				delta_y=delta_y2;
				return max2;
			}

		}

		// this is the case of the maximum inside the boundaries:
		return (coeff1*delta_x*delta_x+coeff2*delta_y*delta_y
			+coeff3*delta_x+coeff4*delta_y
			+coeff5*delta_x*delta_y
			+coeff6)/18.f;
}


/*
fit z=a1*x^2+a2*y^2+a3*x+a4*y+a5*xy+a6
*/
__inline__ float FrifScaleSpace::subpixel2DMax(const float s_0_0, const float s_0_1, const float s_0_2,
	const float s_1_0, const float s_1_1, const float s_1_2,
	const float s_2_0, const float s_2_1, const float s_2_2,
	float& delta_x, float& delta_y){

		// the coefficients of the 2d quadratic function least-squares fit:
		register float tmp1 =        s_0_0 + s_0_2 - 2*s_1_1 + s_2_0 + s_2_2;
		register float coeff1 = 3*(tmp1 + s_0_1 - ((s_1_0 + s_1_2)*2) + s_2_1);
		register float coeff2 = 3*(tmp1 - ((s_0_1+ s_2_1)*2) + s_1_0 + s_1_2 );
		register float tmp2 =                                  s_0_2 - s_2_0;
		register float tmp3 =                         (s_0_0 + tmp2 - s_2_2);
		register float tmp4 =                                   tmp3 -2*tmp2;
		register float coeff3 =                    -3*(tmp3 + s_0_1 - s_2_1);
		register float coeff4 =                    -3*(tmp4 + s_1_0 - s_1_2);
		register float coeff5 =            (s_0_0 - s_0_2 - s_2_0 + s_2_2)*4;
		register float coeff6 = -(s_0_0  + s_0_2 - ((s_1_0 + s_0_1 + s_1_2 + s_2_1)*2) - 5*s_1_1  + s_2_0  + s_2_2)*2;


		// 2nd derivative test:
		register float H_det=4*coeff1*coeff2 - coeff5*coeff5;

		if(H_det==0){
			delta_x=0.f;
			delta_y=0.f;
			return coeff6/18.f;
		}

		if(!(H_det>0.f&&coeff1<0.f)){
			// The maximum must be at the one of the 4 patch corners.

			float tmp_max=coeff3+coeff4+coeff5;
			delta_x=1.f; delta_y=1.f;

			float tmp = -coeff3+coeff4-coeff5;
			if(tmp>tmp_max){
				tmp_max=tmp;
				delta_x=-1.f; delta_y=1.f;
			}
			tmp = coeff3-coeff4-coeff5;
			if(tmp>tmp_max){
				tmp_max=tmp;
				delta_x=1.f; delta_y=-1.f;
			}
			tmp = -coeff3-coeff4+coeff5;
			if(tmp>tmp_max){
				tmp_max=tmp;
				delta_x=-1.f; delta_y=-1.f;
			}
			return (tmp_max+coeff1+coeff2+coeff6)/18.f;
		}

		// this is hopefully the normal outcome of the Hessian test
		delta_x=(2.f*coeff2*coeff3 - coeff4*coeff5)/(-H_det);
		delta_y=(2.f*coeff1*coeff4 - coeff3*coeff5)/(-H_det);


		// TODO: this is not correct, but easy, so perform a real boundary maximum search:
		bool tx=false; bool tx_=false; bool ty=false; bool ty_=false;
		if(delta_x>1.f) tx=true;
		else if(delta_x<-1.f) tx_=true;
		if(delta_y>1.f) ty=true;
		else if(delta_y<-1.f) ty_=true;

		if(tx||tx_||ty||ty_){	//the extrema is not in [-1,1], then search for it
			// get two candidates:
			float delta_x1=0.f, delta_x2=0.f, delta_y1=0.f, delta_y2=0.f;
			if(tx) {
				delta_x1=1.f;
				delta_y1=-(coeff4+coeff5)/(2.f*coeff2);
				if(delta_y1>1.f) delta_y1=1.f; else if (delta_y1<-1.f) delta_y1=-1.f;
			}
			else if(tx_) {
				delta_x1=-1.f;
				delta_y1=-(coeff4-coeff5)/(2.f*coeff2);
				if(delta_y1>1.f) delta_y1=1.f; else if (delta_y1<-1.f) delta_y1=-1.f;
			}
			if(ty) {
				delta_y2=1.f;
				delta_x2=-(coeff3+coeff5)/(2.f*coeff1);
				if(delta_x2>1.f) delta_x2=1.f; else if (delta_x2<-1.f) delta_x2=-1.f;
			}
			else if(ty_) {
				delta_y2=-1.f;
				delta_x2=-(coeff3-coeff5)/(2.f*coeff1);
				if(delta_x2>1.f) delta_x2=1.f; else if (delta_x2<-1.f) delta_x2=-1.f;
			}
			// insert both options for evaluation which to pick
			float max1 = (coeff1*delta_x1*delta_x1+coeff2*delta_y1*delta_y1
				+coeff3*delta_x1+coeff4*delta_y1
				+coeff5*delta_x1*delta_y1
				+coeff6)/18.f;
			float max2 = (coeff1*delta_x2*delta_x2+coeff2*delta_y2*delta_y2
				+coeff3*delta_x2+coeff4*delta_y2
				+coeff5*delta_x2*delta_y2
				+coeff6)/18.f;

			if(max1>max2) {
				delta_x=delta_x1;
				delta_y=delta_y1;
				return max1;
			}
			else{
				delta_x=delta_x2;
				delta_y=delta_y2;
				return max2;
			}

		}

		// this is the case of the maximum inside the boundaries:
		return (coeff1*delta_x*delta_x+coeff2*delta_y*delta_y
			+coeff3*delta_x+coeff4*delta_y
			+coeff5*delta_x*delta_y
			+coeff6)/18.f;
}

void FrifScaleSpace::getBlob(std::vector<cv::KeyPoint>& keypoints)
{
	// make sure keypoints is empty
	keypoints.resize(0);
	keypoints.reserve(2000);

	std::vector<std::vector<cv::Point> > blobPoints;
	blobPoints.resize(layers_);

	//BLOB score
	FrifLayer& l=pyramid_[0];
	l.getBlobPoints(params_.blobThresh, blobPoints[0]);

	const int num=blobPoints[0].size();


	for(int n=0; n < num; n++)
	{
		const cv::Point& point=blobPoints.at(0)[n];

		// first check if it is a maximum:
		if (!isMax2D(point.x, point.y, pyramid_[0].blobScores()))
		{
			continue;
		}

		// let's do the subpixel and float scale refinement:
		register float s_0_0 = l.getBlobScore(point.x-1, point.y-1);
		register float s_1_0 = l.getBlobScore(point.x,   point.y-1);
		register float s_2_0 = l.getBlobScore(point.x+1, point.y-1);
		register float s_2_1 = l.getBlobScore(point.x+1, point.y);
		register float s_1_1 = l.getBlobScore(point.x,   point.y);
		register float s_0_1 = l.getBlobScore(point.x-1, point.y);
		register float s_0_2 = l.getBlobScore(point.x-1, point.y+1);
		register float s_1_2 = l.getBlobScore(point.x,   point.y+1);
		register float s_2_2 = l.getBlobScore(point.x+1, point.y+1);
		float delta_x, delta_y;
		float score = subpixel2DMax(s_0_0, s_0_1, s_0_2,
			s_1_0, s_1_1, s_1_2,
			s_2_0, s_2_1, s_2_2,
			delta_x, delta_y);

		// finally store the detected keypoint:
		if(score>params_.blobThresh)
			keypoints.push_back(cv::KeyPoint(float(point.x)+delta_x, float(point.y)+delta_y, basicSize_, -1, score,0));

	}
}

void FrifScaleSpace::getBlobBlob( std::vector<cv::KeyPoint>& keypoints)
{
	// make sure keypoints is empty
	keypoints.resize(0);
	keypoints.reserve(4000);

	std::vector<std::vector<cv::Point> > blobPoints;
	blobPoints.resize(layers_);

	//BLOB score
	// go through the octaves and intra layers and calculate fast corner scores:
	for(uint8_t i = 0; i<layers_; i++){

		FrifLayer& l=pyramid_[i];
		l.getBlobPoints(params_.blobThresh, blobPoints[i]);
	}

	float x,y,scale,score;
	for(uint8_t i = 0; i<layers_; i++)
	{
		cv::FrifLayer& l=pyramid_[i];
		const int num=blobPoints[i].size();


		for(int n=0; n < num; n++)
		{
			const cv::Point& point=blobPoints.at(i)[n];

			// first check if it is a maximum:
			if (!isMax2D(point.x, point.y, pyramid_[i].blobScores()))
			{
				continue;
			}

			// let's do the subpixel and float scale refinement:
			bool ismax;
			score=refine3DBlobBlob(i,point.x, point.y,x,y,scale,ismax);
			if(!ismax){

				continue;
			}

			// finally store the detected keypoint:
			if(score>params_.blobThresh)
			{

				FrifLayer& layer = pyramid_[i];
				if(layer.suppressLines(point.x, point.y))
				{
					continue;
				}

				keypoints.push_back(cv::KeyPoint(x, y, basicSize_*scale, -1, score,i));
			}
		}
	}
}


/*********************************************************************************************

	FrifLayer

**********************************************************************************************/

// construct a layer
FrifLayer::FrifLayer(const cv::Mat& img, Params& params, float scale, float offset) {
	img_=img;

	// attention: this means that the passed image reference must point to persistent memory
	scale_=scale;
	offset_=offset;

	params_ = params;

	initial();
}

// derive a layer
FrifLayer::FrifLayer(const FrifLayer& layer, int mode, Params& params){
	if(mode==CommonParams::HALFSAMPLE){
		img_.create(layer.img().rows/2, layer.img().cols/2,CV_8U);
		halfsample(layer.img(), img_);
		//resize(layer.img(), img_, img_.size(), CV_INTER_AREA);
		scale_= layer.scale()*2;
		offset_=0.5f*scale_-0.5f;
	}
	else {
		img_.create(2*(layer.img().rows/3), 2*(layer.img().cols/3),CV_8U);
		twothirdsample(layer.img(), img_);
		//resize(layer.img(), img_, img_.size(), CV_INTER_AREA);
		scale_= layer.scale()*1.5f;
		offset_=0.5f*scale_-0.5f;
	}

	params_  = params;

	initial();
}

FrifLayer::~FrifLayer()
{
	img_.release();
	intImg_.release();
	blobScores_.release();
}

__inline__ void FrifLayer::initial()
{

	blobScores_ = cv::Mat::zeros(img_.rows, img_.cols,CV_32FC1);
	calIntegral(img_, intImg_);


	//6n+3,2n+3,2n+1

	hessian_tr_n_ = 2;
	start_ = 3*hessian_tr_n_+1;
	col_end_ = img_.cols-start_;
	row_end_ = img_.rows-start_;

	///////////////////////////////////////////////////
	hessian_tr_n_5_8_ = hessian_tr_n_-1;
	start_5_8_ = 3*hessian_tr_n_5_8_+1;
	col_end_5_8_ = img_.cols-start_5_8_;
	row_end_5_8_ = img_.rows-start_5_8_;
}


/*
// half sampling
inline void FrifLayer::halfsample_sse(const cv::Mat& srcimg, cv::Mat& dstimg){
	const unsigned short leftoverCols = ((srcimg.cols%16)/2);// take care with border...
	const bool noleftover = (srcimg.cols%16)==0; // note: leftoverCols can be zero but this still false...

	// make sure the destination image is of the right size:
	assert(srcimg.cols/2==dstimg.cols);
	assert(srcimg.rows/2==dstimg.rows);

	// mask needed later:
	register __m128i mask = _mm_set_epi32 (0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF);
	// to be added in order to make successive averaging correct:
	register __m128i ones = _mm_set_epi32 (0x11111111, 0x11111111, 0x11111111, 0x11111111);

	// data pointers:
	__m128i* p1=(__m128i*)srcimg.data;
	__m128i* p2=(__m128i*)(srcimg.data+srcimg.cols);
	__m128i* p_dest=(__m128i*)dstimg.data;
	unsigned char* p_dest_char;//=(unsigned char*)p_dest;

	// size:
	const unsigned int size = (srcimg.cols*srcimg.rows)/16;
	const unsigned int hsize = srcimg.cols/16;
	__m128i* p_end=p1+size;
	unsigned int row=0;
	const unsigned int end=hsize/2;
	bool half_end;
	if(hsize%2==0)
		half_end=false;
	else
		half_end=true;
	while(p2<p_end){
		for(unsigned int i=0; i<end;i++){
			// load the two blocks of memory:
			__m128i upper;
			__m128i lower;
			if(noleftover){
				upper=_mm_load_si128(p1);
				lower=_mm_load_si128(p2);
			}
			else{
				upper=_mm_loadu_si128(p1);
				lower=_mm_loadu_si128(p2);
			}

			__m128i result1=_mm_adds_epu8 (upper, ones);
			result1=_mm_avg_epu8 (upper, lower);

			// increment the pointers:
			p1++;
			p2++;

			// load the two blocks of memory:
			upper=_mm_loadu_si128(p1);
			lower=_mm_loadu_si128(p2);
			__m128i result2=_mm_adds_epu8 (upper, ones);
			result2=_mm_avg_epu8 (upper, lower);
			// calculate the shifted versions:
			__m128i result1_shifted = _mm_srli_si128 (result1, 1);
			__m128i result2_shifted = _mm_srli_si128 (result2, 1);
			// pack:
			__m128i result=_mm_packus_epi16 (_mm_and_si128 (result1, mask),
				_mm_and_si128 (result2, mask));
			__m128i result_shifted = _mm_packus_epi16 (_mm_and_si128 (result1_shifted, mask),
				_mm_and_si128 (result2_shifted, mask));
			// average for the second time:
			result=_mm_avg_epu8(result,result_shifted);

			// store to memory
			_mm_storeu_si128 (p_dest, result);

			// increment the pointers:
			p1++;
			p2++;
			p_dest++;
			//p_dest_char=(unsigned char*)p_dest;
		}

		// if we are not at the end of the row, do the rest:
		if(half_end)
		{
			// load the two blocks of memory:
			__m128i upper;
			__m128i lower;
			if(noleftover){
				upper=_mm_load_si128(p1);
				lower=_mm_load_si128(p2);
			}
			else{
				upper=_mm_loadu_si128(p1);
				lower=_mm_loadu_si128(p2);
			}

			__m128i result1=_mm_adds_epu8 (upper, ones);
			result1=_mm_avg_epu8 (upper, lower);

			// increment the pointers:
			p1++;
			p2++;

			// compute horizontal pairwise average and store
			p_dest_char=(unsigned char*)p_dest;
			const unsigned char* result=(unsigned char*)&result1;
			for(unsigned int j=0; j<8; j++){
				*(p_dest_char++)=(*(result+2*j)+*(result+2*j+1))/2;
			}
			//p_dest_char=(unsigned char*)p_dest;
		}
		else
		{
			p_dest_char=(unsigned char*)p_dest;
		}

		if(noleftover){
			row++;
			p_dest=(__m128i*)(dstimg.data+row*dstimg.cols);
			p1=(__m128i*)(srcimg.data+2*row*srcimg.cols);
			//p2=(__m128i*)(srcimg.data+(2*row+1)*srcimg.cols);
			//p1+=hsize;
			p2=p1+hsize;
		}
		else
		{
			const unsigned char* p1_src_char=(unsigned char*)(p1);
			const unsigned char* p2_src_char=(unsigned char*)(p2);
			for(unsigned int k=0; k<leftoverCols; k++)
			{
				unsigned short tmp = p1_src_char[k]+p1_src_char[k+1]+
					p2_src_char[k]+p2_src_char[k+1];
				*(p_dest_char++)=(unsigned char)(tmp/4);

			}
			// done with the two rows:
			row++;
			p_dest=(__m128i*)(dstimg.data+row*dstimg.cols);
			p1=(__m128i*)(srcimg.data+2*row*srcimg.cols);
			p2=(__m128i*)(srcimg.data+(2*row+1)*srcimg.cols);
		}
	}
}
*/

inline void FrifLayer::halfsample_nosse(const cv::Mat& srcimg, cv::Mat& dstimg){

	// make sure the destination image is of the right size:
	assert((srcimg.cols/2)==dstimg.cols);
	assert((srcimg.rows/2)==dstimg.rows);

	// data pointers:
	unsigned char* p1=srcimg.data;		//srcµÚÒ»ÐÐÊ×
	unsigned char* p2=p1+srcimg.cols;	//srcµÚ¶þÐÐÊ×
	unsigned char* p_dest1 = dstimg.data;	//dstµÚÒ»ÐÐÊ×

	unsigned char* p_end=p1+(srcimg.cols*srcimg.rows);

	unsigned int row=0;
	unsigned int row_dest=0;

	while(p2<p_end){

		for(int j = 0; j<srcimg.cols;j+=2){
			const unsigned short A1=*(p1++);
			const unsigned short A2=*(p1++);
			const unsigned short B1=*(p2++);
			const unsigned short B2=*(p2++);

			*(p_dest1++)=(unsigned char)(((A1+A2+B1+B2)/4)&0x00FF);
		}
		// increment row counter:
		row+=2;
		row_dest+=1;

		// reset pointers
		p1=srcimg.data+row*srcimg.cols;
		p2=p1+srcimg.cols;
		p_dest1 = dstimg.data+row_dest*dstimg.cols;
	}
}

/*
inline void FrifLayer::twothirdsample_sse(const cv::Mat& srcimg, cv::Mat& dstimg){
	const unsigned short leftoverCols = ((srcimg.cols/3)*3)%15;// take care with border...

	// make sure the destination image is of the right size:
	assert((srcimg.cols/3)*2==dstimg.cols);
	assert((srcimg.rows/3)*2==dstimg.rows);

	//
	register __m128i mask1 = _mm_set_epi8 (0x80,0x80,0x80,0x80,0x80,0x80,0x80,13,  0x80,10,  0x80,7,   0x80,4,	 0x80,1);
	register __m128i mask2 = _mm_set_epi8 (0x80,0x80,0x80,0x80,0x80,0x80,13,  0x80,10,	0x80,7,   0x80,4,	0x80,1,   0x80);
	register __m128i mask = _mm_set_epi8  (0x80,0x80,0x80,0x80,0x80,0x80,14,  12,  11,  9,   8,   6,   5,   3,   2,   0);
	register __m128i store_mask = _mm_set_epi8 (0,0,0,0,0,0,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80);

	// data pointers:
	unsigned char* p1=srcimg.data;		//srcµÚÒ»ÐÐÊ×
	unsigned char* p2=p1+srcimg.cols;	//srcµÚ¶þÐÐÊ×
	unsigned char* p3=p2+srcimg.cols;	//srcµÚÈýÐÐÊ×
	unsigned char* p_dest1 = dstimg.data;	//dstµÚÒ»ÐÐÊ×
	unsigned char* p_dest2 = p_dest1+dstimg.cols;	//dstµÚ¶þÐÐÊ×
	unsigned char* p_end=p1+(srcimg.cols*srcimg.rows);

	unsigned int row=0;
	unsigned int row_dest=0;
	int hsize = srcimg.cols/15;
	while(p3<p_end){
		for(int i=0; i<hsize; i++){
			// load three rows
			__m128i first = _mm_loadu_si128((__m128i*)p1);
			__m128i second = _mm_loadu_si128((__m128i*)p2);
			__m128i third = _mm_loadu_si128((__m128i*)p3);

			// upper row:
			__m128i upper = _mm_avg_epu8(_mm_avg_epu8(first,second),first);
			__m128i temp1_upper = _mm_or_si128(_mm_shuffle_epi8(upper,mask1),_mm_shuffle_epi8(upper,mask2));
			__m128i temp2_upper=_mm_shuffle_epi8(upper,mask);
			__m128i result_upper = _mm_avg_epu8(_mm_avg_epu8(temp2_upper,temp1_upper),temp2_upper);

			// lower row:
			__m128i lower = _mm_avg_epu8(_mm_avg_epu8(third,second),third);
			__m128i temp1_lower = _mm_or_si128(_mm_shuffle_epi8(lower,mask1),_mm_shuffle_epi8(lower,mask2));
			__m128i temp2_lower=_mm_shuffle_epi8(lower,mask);
			__m128i result_lower = _mm_avg_epu8(_mm_avg_epu8(temp2_lower,temp1_lower),temp2_lower);

			// store:
			if(i*10+16>dstimg.cols){
				_mm_maskmoveu_si128(result_upper, store_mask, (char*)p_dest1);
				_mm_maskmoveu_si128(result_lower, store_mask, (char*)p_dest2);
			}
			else{
				_mm_storeu_si128 ((__m128i*)p_dest1, result_upper);
				_mm_storeu_si128 ((__m128i*)p_dest2, result_lower);
			}

			// shift pointers:
			p1+=15;
			p2+=15;
			p3+=15;
			p_dest1+=10;
			p_dest2+=10;
		}

		// fill the remainder:
		for(unsigned int j = 0; j<leftoverCols;j+=3){
			const unsigned short A1=*(p1++);
			const unsigned short A2=*(p1++);
			const unsigned short A3=*(p1++);
			const unsigned short B1=*(p2++);
			const unsigned short B2=*(p2++);
			const unsigned short B3=*(p2++);
			const unsigned short C1=*(p3++);
			const unsigned short C2=*(p3++);
			const unsigned short C3=*(p3++);

			*(p_dest1++)=(unsigned char)(((4*A1+2*(A2+B1)+B2)/9)&0x00FF);
			*(p_dest1++)=(unsigned char)(((4*A3+2*(A2+B3)+B2)/9)&0x00FF);
			*(p_dest2++)=(unsigned char)(((4*C1+2*(C2+B1)+B2)/9)&0x00FF);
			*(p_dest2++)=(unsigned char)(((4*C3+2*(C2+B3)+B2)/9)&0x00FF);
		}


		// increment row counter:
		row+=3;
		row_dest+=2;

		// reset pointers
		p1=srcimg.data+row*srcimg.cols;
		p2=p1+srcimg.cols;
		p3=p2+srcimg.cols;
		p_dest1 = dstimg.data+row_dest*dstimg.cols;
		p_dest2 = p_dest1+dstimg.cols;
	}
}
*/

inline void FrifLayer::twothirdsample_nosse(const cv::Mat& srcimg, cv::Mat& dstimg){

	// make sure the destination image is of the right size:
	assert((srcimg.cols/3)*2==dstimg.cols);
	assert((srcimg.rows/3)*2==dstimg.rows);

	// data pointers:
	unsigned char* p1=srcimg.data;		//begining of the first row of src
	unsigned char* p2=p1+srcimg.cols;	//begining of the second row of src
	unsigned char* p3=p2+srcimg.cols;	//begining of the third row of src
	unsigned char* p_dest1 = dstimg.data;	//begining of the first row of dst
	unsigned char* p_dest2 = p_dest1+dstimg.cols;	//begining of the second row of dst
	unsigned char* p_end=p1+(srcimg.cols*srcimg.rows);

	unsigned int row=0;
	unsigned int row_dest=0;

	while(p3<p_end){

		for(int j = 0; j<srcimg.cols;j+=3){
			const unsigned short A1=*(p1++);
			const unsigned short A2=*(p1++);
			const unsigned short A3=*(p1++);
			const unsigned short B1=*(p2++);
			const unsigned short B2=*(p2++);
			const unsigned short B3=*(p2++);
			const unsigned short C1=*(p3++);
			const unsigned short C2=*(p3++);
			const unsigned short C3=*(p3++);

			*(p_dest1++)=(unsigned char)(((4*A1+2*(A2+B1)+B2)/9)&0x00FF);
			*(p_dest1++)=(unsigned char)(((4*A3+2*(A2+B3)+B2)/9)&0x00FF);
			*(p_dest2++)=(unsigned char)(((4*C1+2*(C2+B1)+B2)/9)&0x00FF);
			*(p_dest2++)=(unsigned char)(((4*C3+2*(C2+B3)+B2)/9)&0x00FF);
		}


		// increment row counter:
		row+=3;
		row_dest+=2;

		// reset pointers
		p1=srcimg.data+row*srcimg.cols;
		p2=p1+srcimg.cols;
		p3=p2+srcimg.cols;
		p_dest1 = dstimg.data+row_dest*dstimg.cols;
		p_dest2 = p_dest1+dstimg.cols;
	}
}





void FrifLayer::getBlobPoints(float threshold, std::vector<cv::Point>& keypoints)
{
	computeFalogScores();

	int u,v;
	int cols = blobScores_.cols;
	float* data = (float*)blobScores_.data;

	keypoints.resize(0);
	cv::Point p;

	for (v=start_; v<row_end_; v++)
	{
		for (u=start_; u<col_end_; u++)
		{
			if (data[v*cols+u]>=threshold)
			{
				p.x = u;
				p.y = v;
				keypoints.push_back(p);
			}
		}
	}
}


inline float FrifLayer:: getBlobScore(int x, int y)
{
	//assert(x>=0&&y>=0&&x<img_.cols&&y<img_.rows);
	if (!(x>=start_&&y>=start_&&x<col_end_&&y<row_end_))
	{
		return 0.f;
	}

	float* data = (float*)blobScores_.data;
	return data[y*blobScores_.cols+x];
}

inline float FrifLayer:: getBlobScore(float xf, float yf)
{

	int x1 = (int)xf;
	int y1 = (int)yf;
	int x2 = x1+1;
	int y2 = y1+1;

	if (!(x1>=start_&&y1>=start_&&x2<col_end_&&y2<row_end_))
	{
		return 0.f;
	}

	const float rx2=xf-float(x1);
	const float rx1=1.f-rx2;
	const float ry2=yf-float(y1);
	const float ry1=1.f-ry2;



	return rx1*ry1*getBlobScore(x1, y1)+
		rx2*ry1*getBlobScore(x2, y1)+
		rx1*ry2*getBlobScore(x1, y2)+
		rx2*ry2*getBlobScore(x2, y2);

}

inline float FrifLayer:: computeBlobScore_5_8(int x, int y)
{
	//assert(x>=0&&y>=0&&x<img_.cols&&y<img_.rows);
	if (!(x>=start_5_8_&&y>=start_5_8_&&x<col_end_5_8_&&y<row_end_5_8_))
	{
		return 0.f;
	}

	float score = 0.f;
	int int_cols = intImg_.cols;
	int int_rows = intImg_.rows;

	int n = hessian_tr_n_5_8_;

	int s1 = (2*n+3)*(6*n+3);
	int s2 = (2*n+3)*(2*n+1);
	int s3 = s1;
	int s4 = s2;

	float total_weight = fabs(s2*(-2.f));


	float weight1 = 1.f/total_weight;
	float weight2 = -3.f/total_weight;
	float weight3 = weight1;
	float weight4 = weight2;


	//4n+3
	int box1_rw = n+1;
	int box1_rh = 3*n+1;
	int box2_rw = n+1;
	int box2_rh = n;
	int box3_rw = 3*n+1;
	int box3_rh = n+1;
	int box4_rw = n;
	int box4_rh = n+1;

	int* int_data = (int*)intImg_.data;
	float box1, box2, box3, box4;

	float* data = (float*)blobScores_.data;
	int blob_step = blobScores_.step /sizeof(float);

	box1 =  ( int_data[(y-box1_rh)*int_cols + (x-box1_rw)]   + int_data[(y+box1_rh+1)*int_cols + (x+box1_rw+1)]
	- int_data[(y-box1_rh)*int_cols + (x+box1_rw+1)] - int_data[(y+box1_rh+1)*int_cols + (x-box1_rw)]  ) * weight1;

	box2 =  ( int_data[(y-box2_rh)*int_cols + (x-box2_rw)]   + int_data[(y+box2_rh+1)*int_cols + (x+box2_rw+1)]
	- int_data[(y-box2_rh)*int_cols + (x+box2_rw+1)] - int_data[(y+box2_rh+1)*int_cols + (x-box2_rw)]  ) * weight2;

	box3 =  ( int_data[(y-box3_rh)*int_cols + (x-box3_rw)]   + int_data[(y+box3_rh+1)*int_cols + (x+box3_rw+1)]
	- int_data[(y-box3_rh)*int_cols + (x+box3_rw+1)] - int_data[(y+box3_rh+1)*int_cols + (x-box3_rw)]  ) * weight3;

	box4 =  ( int_data[(y-box4_rh)*int_cols + (x-box4_rw)]   + int_data[(y+box4_rh+1)*int_cols + (x+box4_rw+1)]
	- int_data[(y-box4_rh)*int_cols + (x+box4_rw+1)] - int_data[(y+box4_rh+1)*int_cols + (x-box4_rw)]  ) * weight4;

	score = fabs(box1+box2+box3+box4);

	return score;
}

inline float FrifLayer:: computeBlobScore_5_8(float xf, float yf)
{

	int x1 = (int)xf;
	int y1 = (int)yf;
	int x2 = x1+1;
	int y2 = y1+1;

	if (!(x1>=start_5_8_&&y1>=start_5_8_&&x2<col_end_5_8_&&y2<row_end_5_8_))
	{
		return 0.f;
	}


	const float rx2=xf-float(x1);
	const float rx1=1.f-rx2;
	const float ry2=yf-float(y1);
	const float ry1=1.f-ry2;

	return rx1*ry1*computeBlobScore_5_8(x1, y1)+
		rx2*ry1*computeBlobScore_5_8(x2, y1)+
		rx1*ry2*computeBlobScore_5_8(x1, y2)+
		rx2*ry2*computeBlobScore_5_8(x2, y2);

}

inline void FrifLayer::computeFalogScores()
{
	int u,v;

	int n = hessian_tr_n_;

	int s1 = (2*n+3)*(6*n+3);
	int s2 = (2*n+3)*(2*n+1);
	int s3 = s1;
	int s4 = s2;

	//for normalization
	float total_weight = fabs(s2*(-2.f));

	//for different rectangle filter
	float weight1 = 1.f/total_weight;
	float weight2 = -3.f/total_weight;
	float weight3 = weight1;
	float weight4 = weight2;

	//4n+3
	int box1_rw = n+1;
	int box1_rh = 3*n+1;
	int box2_rw = n+1;
	int box2_rh = n;
	int box3_rw = 3*n+1;
	int box3_rh = n+1;
	int box4_rw = n;
	int box4_rh = n+1;

	int* int_data = (int*)intImg_.data;
	int int_step = intImg_.step/sizeof(int);
	float* data = (float*)blobScores_.data;
	int blob_step = blobScores_.step /sizeof(float);

	int* locs1[4];
	int* locs2[4];
	int* locs3[4];
	int* locs4[4];

	locs1[0] = int_data-box1_rh*int_step-box1_rw;
	locs1[1] = int_data-box1_rh*int_step+box1_rw+1;
	locs1[2] = int_data+(box1_rh+1)*int_step+box1_rw+1;
	locs1[3] = int_data+(box1_rh+1)*int_step-box1_rw;

	locs2[0] = int_data-box2_rh*int_step-box2_rw;
	locs2[1] = int_data-box2_rh*int_step+box2_rw+1;
	locs2[2] = int_data+(box2_rh+1)*int_step+box2_rw+1;
	locs2[3] = int_data+(box2_rh+1)*int_step-box2_rw;

	locs3[0] = int_data-box3_rh*int_step-box3_rw;
	locs3[1] = int_data-box3_rh*int_step+box3_rw+1;
	locs3[2] = int_data+(box3_rh+1)*int_step+box3_rw+1;
	locs3[3] = int_data+(box3_rh+1)*int_step-box3_rw;

	locs4[0] = int_data-box4_rh*int_step-box4_rw;
	locs4[1] = int_data-box4_rh*int_step+box4_rw+1;
	locs4[2] = int_data+(box4_rh+1)*int_step+box4_rw+1;
	locs4[3] = int_data+(box4_rh+1)*int_step-box4_rw;


	int int_row_offset, blob_row_offset,offset;
	float box1, box2, box3, box4;
	for (v=start_; v<row_end_; v++)
	{
		int_row_offset = v*int_step;
		blob_row_offset = v*blob_step;
		for (u=start_; u<col_end_; u++)
		{
			offset = int_row_offset+u;
			box1  =  (locs1 [0][offset] + locs1 [2][offset] - locs1 [1][offset] - locs1 [3][offset]) * weight1;
			box2  =  (locs2 [0][offset] + locs2 [2][offset] - locs2 [1][offset] - locs2 [3][offset]) * weight2;
			box3  =  (locs3 [0][offset] + locs3 [2][offset] - locs3 [1][offset] - locs3 [3][offset]) * weight3;
			box4  =  (locs4 [0][offset] + locs4 [2][offset] - locs4 [1][offset] - locs4 [3][offset]) * weight4;


			data[blob_row_offset+u] = fabs(box1+box2+box3+box4);
		}
	}

	return;
}

__inline__ bool FrifLayer::suppressLines(const int x, const int y)
{
	if (!params_.supLine)
		return false;

	int* int_data = (int*)intImg_.data;
	int int_step = intImg_.step/sizeof(int);

	int radius = 3;	// of window
	int kernel_radius = 2;//for smooth
	int grad_radius = 1;
	float kernel_area = float(2*kernel_radius+1)*(2*kernel_radius+1);

	int margin = radius+kernel_radius+grad_radius;

	if (x<margin || y<margin || x>img_.cols-1-margin || y>img_.rows-1-margin)
		return true;

	int* kernel_locs[4];
	kernel_locs[0] = int_data-kernel_radius*int_step-kernel_radius;
	kernel_locs[1] = int_data-kernel_radius*int_step+kernel_radius+1;
	kernel_locs[2] = int_data+(kernel_radius+1)*int_step+kernel_radius+1;
	kernel_locs[3] = int_data+(kernel_radius+1)*int_step-kernel_radius;

	float LxLx=0.f;
	float LyLy=0.f;
	float LxLy=0.f;
	for( int v = y - radius; v <= y + radius; v += 1 )
	{
		for(int u = x - radius; u <= x + radius; u += 1)
		{
			int oft1  = v*int_step+u+grad_radius;
			int oft_1 = v*int_step+u-grad_radius;
			float Lx = ((kernel_locs[0][oft1] + kernel_locs[2][oft1] - kernel_locs[1][oft1] - kernel_locs[3][oft1])
				-  (kernel_locs[0][oft_1] + kernel_locs[2][oft_1] - kernel_locs[1][oft_1] - kernel_locs[3][oft_1]))/kernel_area;

			oft1  = (v+grad_radius)*int_step+u;
			oft_1 = (v-grad_radius)*int_step+u;
			float Ly = ((kernel_locs[0][oft1]  + kernel_locs[2][oft1]  - kernel_locs[1][oft1]  - kernel_locs[3][oft1])
				- (kernel_locs[0][oft_1] + kernel_locs[2][oft_1] - kernel_locs[1][oft_1] - kernel_locs[3][oft_1]))/kernel_area;

			LxLx += Lx*Lx;
			LyLy += Ly*Ly;
			LxLy += Lx*Ly;
		}
	}

	float det = LxLx*LyLy-LxLy*LxLy;
	float tr = LxLx+LyLy;
	if( tr*tr >= params_.lineThresh*det )
		return true;

	return false;
}
