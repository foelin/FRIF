#include "utils.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{

	const char* command = "";

	char* img_file = NULL;
	const char* kp_file = "test.kp";
	const char* des_file = "test.des";


	TYPE type = DETECTION_AND_DESCRIPTION;
	Params params;

	int counter=0;
	while( ++counter < argc )
	{
		if( !strcmp("-img", argv[counter] ))
		{
			img_file = argv[++counter];
			continue;
		}

		if( !strcmp("-kp", argv[counter] ))
		{
			kp_file = argv[++counter];
			continue;
		}


		if( !strcmp("-des", argv[counter] ))
		{
			des_file = argv[++counter];
			continue;
		}

		if( !strcmp("-type", argv[counter] ))
		{
			type = (TYPE)atoi(argv[++counter]);
			continue;
		}

		if( !strcmp("-blobThresh", argv[counter] ))
		{
			params.blobThresh = (float)atof(argv[++counter]);
			continue;
		}

		if( !strcmp("-border", argv[counter] ))
		{
			params.border = atoi(argv[++counter]);
			continue;
		}

		if( !strcmp("-supLine", argv[counter] ))
		{
			params.supLine = atoi(argv[++counter]);
			continue;
		}

		if( !strcmp("-lineThresh", argv[counter] ))
		{
			params.lineThresh = (float)atof(argv[++counter]);
			continue;
		}

		//wrong argument
		cerr << "Invalid command line argument: \"" << argv[counter] <<"\""<< endl;
		return -1;
	}

	if(!img_file)
	{
		cerr << "Image file has not been specified!"<<endl;
		return -1;
	}

	Mat img = cv::imread(img_file, IMREAD_GRAYSCALE);
	vector<KeyPoint> kpts;
	Mat dess;

	cv::BRISK test;

	switch (type)
	{
	case DETECTION:
		{

			computeKp(img, kpts, params);
			writeKp(kp_file, kpts);

		}
		break;

	case DESCRIPTION:
		{

			if(!readKp(kp_file, kpts))
				return -1;

			computeDes(img, kpts, dess, params);
			writeDes(des_file, kpts, dess);
		}
		break;

	case DETECTION_AND_DESCRIPTION:
		{

			computeKpAndDes(img, kpts, dess, params);
			writeKp(kp_file, kpts);
			writeDes(des_file, kpts, dess);
		}
		break;

	default:
		cerr << "Undefined -type "<<type<<" value!"<<endl;
		return -1;

	}

	return 0;
}
