#include "utils.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/video/tracking.hpp>
#include <vector>


//#include "tracking.hpp"
using namespace cv;

//avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)

#ifndef COLORMODULE_H_INCLUDE
#define COLORMODULE_H_INCLUDE

namespace colo {


class colorModule {




public:
	Mat patch;

	cvPatch track(Mat frame, std::vector<cvPatch> patches, int bins,
			int mode);
	cvPatch computeDistance(std::vector<cvPatch> hist_candlist, Mat histModel);
	std::vector<cvPatch> computePatchHist(std::vector<cvPatch> patches,
			int mode, int bins);

	Mat calculateChannelR(Mat model, int bins);
	Mat calculateChannelG(Mat model, int bins);
	Mat calculateChannelB(Mat model, int bins);
	Mat calculateChannelH(Mat model, int bins);
	Mat calculateChannelS(Mat model, int bins);
	Mat calculateChannelGray(Mat model, int bins);
	void visualizeHist(Mat histMat, int bins);
private:
	Mat hist;

};

}




#endif
