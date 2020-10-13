/*
 * fuseModule.hpp
 *
 *  Created on: May 8, 2020
 *      Author: mohammad
 */
#include <opencv2/opencv.hpp>
#include <vector>
#include "gradientModule.hpp"
#include "colorModule.hpp"
#include "opencv2/imgcodecs.hpp"
#include <math.h>
#include "ShowManyImages.hpp"

#ifndef SRC_FUSEMODULE_HPP_
#define SRC_FUSEMODULE_HPP_
namespace fuse {


class fuseModule {

public:
	cvPatch track(Mat model, vector<cvPatch> patches, int bins, int nbin,
			int mode);
	cvPatch computeFusedScore(vector<cvPatch> hist_candlist,
			vector<cvPatch> grad_candlist, Mat histModel, Mat gradModel);
private:

};


}



#endif /* SRC_FUSEMODULE_HPP_ */
