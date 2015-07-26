#ifndef UTILS_H
#define UTILS_H

#include<opencv2/opencv.hpp>

#include <vector>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include<fstream>
#include<time.h>
#include <sys/time.h>
#include "Common.h"

using namespace std;
using namespace cv;


enum TYPE{DETECTION=0, DESCRIPTION, DETECTION_AND_DESCRIPTION};
const float FRIF_PATCH_SIZE = 31.f;		//include sigma
const float FRIF_BASE_SIZE = 12.f*0.6f;


void computeKp(const Mat& img, vector<KeyPoint>& kpts, Params& params);
void computeDes(const Mat& img, vector<KeyPoint>& kpts, cv::Mat& dess, Params& params);
void computeKpAndDes(const Mat& img, vector<KeyPoint>& kpts, cv::Mat& dess, Params& params);

bool readKp(const char* kp_file, vector<KeyPoint>& kpts);
bool writeKp(const char* kp_file, vector<KeyPoint>& kpts);
bool writeDes(const char* des_file, vector<KeyPoint>& kpts, Mat& dess);
void writeKp(ofstream& kp_of, vector<KeyPoint>& kpts, int dim);
void writeDes(ofstream& des_of, vector<KeyPoint>& kpts, Mat_<uchar>& dess);

bool compareKeypoint(KeyPoint i, KeyPoint j);
bool compareFrifPair(FrifPair i, FrifPair j);
void calIntegral(const cv::Mat& src, cv::Mat& dst);


inline float angle2radian(float angle)
{
	return float(CV_PI * angle / 180.f);
}
inline float radian2angle(float radian)
{
	return float(radian / CV_PI * 180.f);
}

inline bool isOutOfBound(const float minX, const float minY,
	const float maxX, const float maxY, const KeyPoint& keyPt){
		const Point2f& pt = keyPt.pt;
		return (pt.x < minX) || (pt.x >= maxX) || (pt.y < minY) || (pt.y >= maxY);
}

#endif
////////////////////////////////////////////////////////
