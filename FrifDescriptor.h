/*
FRIF: Fast Robust Invariant Feature
Reference:
[1] Zhenhua Wang, Bin Fan and Fuchao Wu, FRIF: Fast Robust Invariant Feature, in
British Machine Vision Conference, 2013

FRIF is a free software.
Current implementation is based on the source code of BRISK provided
by Stefan Leutenegger et al.

You can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

You should have received a copy of the GNU General Public License
along with FRIF.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _FRIF_DESCRIPTOR_H_
#define _FRIF_DESCRIPTOR_H_

#include <opencv2/opencv.hpp>

#include <emmintrin.h>

#include "Common.h"
#include <vector>

#ifndef CV_PI
#define CV_PI 3.141592653589793
#endif

using namespace std;

namespace cv{


#define __inline__ inline


	class CV_EXPORTS FrifDescriptorExtractor : public cv::DescriptorExtractor {
	public:
		// create a descriptor with standard pattern
		FrifDescriptorExtractor();

		// custom setup
		FrifDescriptorExtractor(Params& params);

		virtual ~FrifDescriptorExtractor();

		// call this to generate the kernel:
		// circle of radius r (pixels), with n points;
		// short pairings with dMax, long pairings with dMin
		void generateKernel();

		// TODO: implement read and write functions
		//virtual void read( const cv::FileNode& );
		//virtual void write( cv::FileStorage& ) const;

		int descriptorSize() const;
		int descriptorType() const;

		virtual void computeImpl(const Mat& image, std::vector<KeyPoint>& keypoints,
			Mat& descriptors) const;

		virtual void compute(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const{
			computeImpl(image,keypoints,descriptors);
		}
		//}

		void computeDomiOrien(const Mat& image, const Mat& integral, std::vector<KeyPoint>& keypoints, std::vector<int>& kscales) const;
		void computeFrif(const Mat& image, const Mat& integral,std::vector<KeyPoint>& keypoints, std::vector<int>& kscales, Mat& descriptors) const;

	protected:
		__inline__ int smoothedIntensity(const cv::Mat& image,
			const cv::Mat& integral,const float key_x,
			const float key_y, const unsigned int scale,
			const unsigned int rot, const unsigned int point) const;

		__inline__ int smoothedIntensity(const cv::Mat& image,
			const cv::Mat& integral,const float key_x,
			const float key_y, const unsigned int scale,
			const unsigned int rot, const unsigned int point, const float delta_x, const float delta_y) const;


		__inline__ int smoothedIntensity_fast(const cv::Mat& image,
			const cv::Mat& integral,const float key_x,
			const float key_y, const unsigned int scale,
			const unsigned int rot, const unsigned int point) const;

		__inline__ int smoothedIntensity_fast(const cv::Mat& image,
			const cv::Mat& integral,const float key_x,
			const float key_y, const unsigned int scale,
			const unsigned int rot, const unsigned int point, const float delta_x, const float delta_y) const;

		// pattern properties
		FrifPatternPoint* patternPoints_; 	//[i][rotation][scale]
		float* scaleList_; 					// lists the scaling per scale index [scale]
		unsigned int* sizeList_; 			// lists the total pattern size per scale index [scale]

		static const unsigned int noPatternPoints_; 	// total number of collocation points
		static const unsigned int scales_;	// scales discretization
		static const float scalerange_; 	// span of sizes 40->4 Octaves - else, this needs to be adjusted...
		static const unsigned int noRot_;	// discretization of the rotation look-up
		static const unsigned int strings_;				// number of byes the descriptor consists of
		static const unsigned int noShortPairs_; 		// number of shortParis
		static const unsigned int noLongPairs_; 		// number of longParis
		static const unsigned int noTotalPairs_;

		vector<FrifPair> frifPairs_;	// pairs

		float* samplePos_;				//position of local samping points around each pattern location

		//parameters
		Params params_;
	};
}

#endif
