#ifndef _FRIF_DETECTOR_H_
#define _FRIF_DETECTOR_H_

#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <emmintrin.h>

#include "Common.h"


#define __inline__ inline
#define halfsample halfsample_nosse
#define twothirdsample twothirdsample_nosse

namespace cv{



// a layer in the FRIF detector pyramid
class CV_EXPORTS FrifLayer
{
public:
	// constructor arguments
	struct CV_EXPORTS CommonParams
	{
		static const int HALFSAMPLE = 0;
		static const int TWOTHIRDSAMPLE = 1;
	};
	// construct a base layer
	FrifLayer(const cv::Mat& img, Params& params, float scale=1.0f, float offset=0.0f);
	// derive a layer
	FrifLayer(const FrifLayer& layer, int mode, Params& params);

	~FrifLayer();

	void getBlobPoints(float threshold, std::vector<cv::Point>& keypoints);
	inline float getBlobScore(int x, int y);
	inline float getBlobScore(float xf, float yf);
	inline float computeBlobScore_5_8(int x, int y);
	inline float computeBlobScore_5_8(float xf, float yf);

	inline void  computeFalogScores();
	__inline__ bool suppressLines(const int x, const int y);

	// accessors
	inline const cv::Mat& img() const {return img_;}
	inline const cv::Mat& blobScores() const {return blobScores_;}
	inline float scale() const {return scale_;}
	inline float offset() const {return offset_;}

	// half sampling
	//static inline void halfsample_sse(const cv::Mat& srcimg, cv::Mat& dstimg);
	// two third sampling
	//static inline void twothirdsample_sse(const cv::Mat& srcimg, cv::Mat& dstimg);

	// half sampling
	static inline void halfsample_nosse(const cv::Mat& srcimg, cv::Mat& dstimg);
	// two third sampling
	static inline void twothirdsample_nosse(const cv::Mat& srcimg, cv::Mat& dstimg);

private:

	__inline__ void initial();

	// the image
	cv::Mat img_;

	float scale_;
	float offset_;
	Params params_;
	cv::Mat blobScores_;	//dob,doo,lap,log

	//integral image
	cv::Mat intImg_;

	int hessian_tr_n_;
	int start_;
	int row_end_;
	int col_end_;
	int hessian_tr_n_5_8_;

	int start_5_8_;
	int row_end_5_8_;
	int col_end_5_8_;
	int area_in_5_8_;
	int area_out_5_8_;

};

class CV_EXPORTS FrifScaleSpace
{
public:
	// construct telling the octaves number:
	FrifScaleSpace(const Params& params);
	~FrifScaleSpace();

	// construct the image pyramids
	void constructPyramid(const cv::Mat& image);

	// get Keypoints
	void getKeypoints(std::vector<cv::KeyPoint>& keypoints);

protected:
	__inline__ bool isMax2D(const int x_layer, const int y_layer, const cv::Mat& scores);
	// 1D (scale axis) refinement:
	__inline__ float refine1D(const float s_05,
		const float s0, const float s05, float& max); // around octave
	__inline__ float refine1D_1(const float s_05,
		const float s0, const float s05, float& max); // around intra
	__inline__ float refine1D_2(const float s_05,
		const float s0, const float s05, float& max); // around octave 0 only

	// 2D maximum refinement:
	__inline__ float subpixel2DMax(const int s_0_0, const int s_0_1, const int s_0_2,
		const int s_1_0, const int s_1_1, const int s_1_2,
		const int s_2_0, const int s_2_1, const int s_2_2,
		float& delta_x, float& delta_y);

	__inline__ float subpixel2DMax(const float s_0_0, const float s_0_1, const float s_0_2,
		const float s_1_0, const float s_1_1, const float s_1_2,
		const float s_2_0, const float s_2_1, const float s_2_2,
		float& delta_x, float& delta_y);

	__inline__ float refine3DBlobBlob(const uint8_t layer,
		const int x_layer, const int y_layer,
		float& x, float& y, float& scale,  bool& ismax);


	void getBlob(std::vector<cv::KeyPoint>& keypoints);
	void getBlobBlob(std::vector<cv::KeyPoint>& keypoints);
	// the image pyramids:
	uint8_t layers_;
	std::vector<cv::FrifLayer> pyramid_;


	// some constant parameters:
	static const float safetyFactor_;
	static const float basicSize_;

private:
	Params params_;
};


// wrapping class for the common interface
class CV_EXPORTS FrifFeatureDetector : public FeatureDetector
{
public:
	FrifFeatureDetector(Params& params);
	//~FastSseFeatureDetector();

	void setThresh(float thresh);

    virtual void detect( const Mat&  image,
    std::vector<KeyPoint>& keypoints, const Mat&  mask=cv::Mat());

protected:
	// also this should in fact be protected...:
	virtual void detectImpl( const cv::Mat& image,
		std::vector<cv::KeyPoint>& keypoints,
		const cv::Mat& mask=cv::Mat() ) const;




	//parameters
	Params params_;
};

}

#endif
