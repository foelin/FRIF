#ifndef COMMON_H
#define COMMON_H

typedef struct _Params{

	//for detector
	int border;
	float blobThresh;
	int octaves;
	int supLine;
	float lineThresh;
	int kpNum;


	//for descriptor
	int lsRadius;	//local sample radius for pattern
	int lsNum;
	int patchSize;

	_Params()
	{
		border = 50;
		blobThresh = 40.f;
		octaves = 4;
		lsNum = 4;
		lsRadius = 3;
		patchSize = 31;
		lineThresh = 30.f;
		supLine = 10;
	}

	_Params& operator=(const _Params& params)
	{
		this->border = params.border;
		this->blobThresh = params.blobThresh;
		this->octaves = params.octaves;
		this->lsRadius = params.lsRadius;
		this->lsNum = params.lsNum;
		this->patchSize = params.patchSize;
		this->supLine = params.supLine;
		this->lineThresh = params.lineThresh;
		return (*this);
	}
}Params;


struct FrifPatternPoint{
	float x;         // x coordinate relative to center
	float y;         // x coordinate relative to center
	float sigma;     // Gaussian smoothing sigma
};

struct FrifPair{
	unsigned int i;  // index of the first pattern point
	unsigned int j;  // index of other pattern point
	int weighted_dx; // 1024.0/dx
	int weighted_dy; // 1024.0/dy
	float dist_sq;
};

#endif
