/*
 * gradientModule.hpp
 *
 *  Created on: May 8, 2020
 *      Author: mohammad
 */
#include "opencv2/opencv.hpp"
#include <opencv2/video/tracking.hpp>
#include<vector>
#include"utils.hpp"

#ifndef SRC_GRADIENTMODULE_HPP_
#define SRC_GRADIENTMODULE_HPP_

//#include "tracking.hpp"
using namespace cv;

//avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)

namespace grad {


class gradientModule {

public:
	cvPatch track(Mat model, std::vector<cvPatch> patches, int bins);
	Mat calculateGradient(Mat model, int bin);

	std::vector<cvPatch> computePatchGradient(std::vector<cvPatch> patches,
			int bin);
	cvPatch compareGradients(std::vector<cvPatch> grad_candlist, Mat gradModel);

private:

};

}




#endif /* SRC_GRADIENTMODULE_HPP_ */
