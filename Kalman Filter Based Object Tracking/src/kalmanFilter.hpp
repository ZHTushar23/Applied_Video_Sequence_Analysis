#include "opencv2/opencv.hpp"
#include <opencv2/video/tracking.hpp>
#include<vector>
//#include "tracking.hpp"
using namespace cv;

//avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)

#ifndef KALMANFILTER_H_INCLUDE
#define KALMANFILTER_H_INCLUDE

namespace kl {
class kalmanFilter {
public:
	kalmanFilter(int type);
	void applyKalman(Point center);
	void initialize_matrices();
	void initialize_filter();
	std::vector<Point> predicted_trajectory();
	std::vector<Point> measurement_trajectory();
	std::vector<Point> corrected_trajectory();
	Point getCurrentPredictedCenter();
	Point getCurrentCorrectedCenter();
	KalmanFilter kf;

	//ADD ADITIONAL METHODS HERE
	//...
private:
	int mat_size;
	int filter_type;
	std::vector<Point> predictedPoints;
	std::vector<Point> correctedPoints;
	std::vector<Point> measurementPoints;
	Point predicted_center;
	Point corrected_center;


	Mat A, Q, H, P, R, Zk, Xk;
	int Z_size;
	bool flag = true;

};
//end of class bgs
}



#endif
