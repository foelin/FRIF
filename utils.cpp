
#include "utils.h"
#include "FrifDetector.h"
#include "FrifDescriptor.h"
#include <ctime>
#include <fstream>

void computeKp(const Mat& img, vector<KeyPoint>& kpts, Params& params)
{
	cv::FrifFeatureDetector* detector = new cv::FrifFeatureDetector(params);

     timeval start, end;
    gettimeofday(&start,NULL);


	vector<KeyPoint> all_kpts;


	detector->detect(img, all_kpts);

	int all_kpts_num = all_kpts.size();
	kpts.reserve(all_kpts_num);
	for (int i=0; i < all_kpts_num; i++ )
	{
		KeyPoint& kpt = all_kpts[i];
		// Reject points too close to the border, round coordinates to integers
		int x = cvRound(kpt.pt.x);
		int y = cvRound(kpt.pt.y);
		if (x<params.border || x>img.cols-1-params.border || y<params.border || y>img.rows-1-params.border)
			continue;
		kpts.push_back(kpt);
	}

    gettimeofday(&end,NULL);
    cout<<"Detection: "<<kpts.size()<<" features, "
    <<(end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec<<" microseconds"<<endl;

	delete detector;
}

void computeDes(const Mat& img, vector<KeyPoint>& kpts, cv::Mat& dess, Params& params)
{
	cv::FrifDescriptorExtractor* des_extractor = new cv::FrifDescriptorExtractor(params);


	 timeval start, end;
    gettimeofday(&start,NULL);

    des_extractor->compute(img, kpts, dess);

    gettimeofday(&end,NULL);
    cout<<"Description: " << kpts.size() << " features, "
    <<(end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec<<" microseconds"<<endl;

	delete des_extractor;

}

void computeKpAndDes(const Mat& img, vector<KeyPoint>& kpts, cv::Mat& dess, Params& params)
{

	cv::FrifFeatureDetector* detector = new cv::FrifFeatureDetector(params);
	cv::FrifDescriptorExtractor* des_extractor = new cv::FrifDescriptorExtractor(params);


    timeval start, end;
    gettimeofday(&start,NULL);


	vector<KeyPoint> all_kpts;
	detector->detect(img, all_kpts);
	int all_kpts_num = all_kpts.size();
	kpts.reserve(all_kpts_num);
	for (int i=0; i < all_kpts_num; i++ )
	{
		KeyPoint& kpt = all_kpts[i];
		// Reject points too close to the border, round coordinates to integers
		int x = cvRound(kpt.pt.x);
		int y = cvRound(kpt.pt.y);
		if (x<params.border || x>img.cols-1-params.border || y<params.border || y>img.rows-1-params.border)
			continue;

		kpts.push_back(kpt);
	}

	des_extractor->compute(img, kpts,dess);

    gettimeofday(&end,NULL);
    cout<<"Detection+Description: "<<kpts.size()<<" features, "
    <<(end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec<<" microseconds"<<endl;


	delete detector;
	delete des_extractor;
}

//Using lowe's sift format
//	num dim
//  y x scale orien
bool readKp(const char* kp_file, vector<KeyPoint>& kpts)
{
	ifstream file(kp_file);
	if(!file.is_open())
	{
		cerr<<"Can NOT open file "<<kp_file<<endl;
		return false;
	}

	int num;
	int dim;
	file >> num >> dim;
	kpts.clear();
	kpts.reserve(num);
	for(int i=0; i<num ; i++)
	{
		float x,y,scale,ori,size,angle;
		file >> y >> x >> scale >> ori;
		angle = radian2angle(ori);
		size = scale*FRIF_BASE_SIZE;
		KeyPoint kp(Point2f(x,y),size,angle, 0, 0, -1);
		kpts.push_back(kp);
	}

	file.close();
	return true;
}

bool writeKp(const char* kp_file, vector<KeyPoint>& kpts)
{
	// Open the file.
	ofstream kp_of(kp_file);
	if (!kp_of.is_open())
	{
		cerr<<"Invalid descriptor file name: "<<kp_file<<endl;
		return false;
	}

	writeKp(kp_of, kpts, 0);
	kp_of.close();

	return true;
}

bool writeDes(const char* des_file, vector<KeyPoint>& kpts, Mat& dess)
{
	ofstream des_of(des_file);
	if(!des_of.is_open())
	{
		cerr<<"Can NOT open file "<<des_file<<endl;
		return false;
	}

	writeDes(des_of, kpts,(Mat_<uchar>&)dess);
	des_of.close();

	return true;
}

void writeKp(ofstream& kp_of, vector<KeyPoint>& kpts, int dim)
{
	int num = kpts.size();

	kp_of<<num<<" "<<dim<<endl;
	for (int i=0; i<num; i++)
	{
		KeyPoint& kp = kpts[i];
		float ori = angle2radian(kp.angle);
		float scale = kp.size / FRIF_BASE_SIZE;
		kp_of	<< setiosflags(ios::fixed) << setprecision(2)
			<< kp.pt.y << " " << kp.pt.x << " " << setiosflags(ios::fixed) << setprecision(4)
			<< scale <<" "<< ori <<endl;
	}

}

void writeDes(ofstream& des_of, vector<KeyPoint>& kpts, Mat_<uchar>& dess)
{
	int num = kpts.size();
	int dim = dess.cols;

	int step = dess.step1();
	uchar* dess_data = (uchar*)dess.data;



	des_of << num << " " <<dim<<endl;
	for(int i=0; i<num ;i++)
	{
		KeyPoint& kp = kpts[i];
		float ori = angle2radian(kp.angle);
		float scale = kp.size / FRIF_BASE_SIZE;
		des_of	<< setiosflags(ios::fixed) << setprecision(2)
			<< kp.pt.y << " " << kp.pt.x << " " << setiosflags(ios::fixed) << setprecision(4)
			<< scale << " "<< ori <<endl;

		for(int j=0; j<dim; j++)
		{
			des_of << (int)dess_data[i*step+j] << " ";

		}
		des_of <<endl;
	}
}

bool compareKeypoint(KeyPoint i, KeyPoint j)
{
	return fabs(i.response) > fabs(j.response);
}

bool compareFrifPair(FrifPair i, FrifPair j)
{
	return i.dist_sq > j.dist_sq;
}

void calIntegral(const cv::Mat& src, cv::Mat& dst)
{
	uchar* src_data = (uchar*)src.data;
	int src_step = src.step/sizeof(uchar);

	int dst_cols = src.cols+1;
	int dst_rows = src.rows+1;
	//dst = cvCreateMat(dst_rows, dst_cols, CV_32SC1);
	dst.create(dst_rows, dst_cols, CV_32SC1);
	int* dst_data = (int*)dst.data;

	int dst_step = dst.step/sizeof(int);

	int u,v;
	for (u=0; u<dst_cols; u++)
		dst_data[u] = 0;
	for (v=1; v<dst_rows; v++)
		dst_data[v*dst_step] = 0;

	for (v=1; v<dst_rows; v++)
	{
		int row_sum = 0;
		for (u=1; u<dst_cols; u++)
		{
			row_sum += src_data[(v-1)*src_step+(u-1)];
			dst_data[v*dst_step+u] = dst_data[(v-1)*dst_step+u] + row_sum;
		}
	}
}

