/*
 * kalmanFilter.cpp
 *
 *  Created on: Apr 7, 2020
 *      Author: mohammad
 */


#include <opencv2/video/tracking.hpp>
#include "kalmanFilter.hpp"
#include<vector>
//#include "tracking.hpp"
using namespace std;
using namespace cv;
using namespace kl;

kalmanFilter::kalmanFilter(int type) {
	
	Z_size = 2;
	filter_type = type;
	if (type == 1) {
		mat_size = 4;
	} else if (type == 2) {
		mat_size = 6;
	} else {
		cout << "Please enter 1 or 2" << endl;
	}
	
	
}

void kalmanFilter::initialize_matrices() {

	//Transition Matrix A
	A = Mat::zeros(Size(mat_size, mat_size), CV_32F);
	setIdentity(A);
	//Process noise covariance Matrix Q
	Q = Mat::zeros(Size(mat_size, mat_size), CV_32F);
	setIdentity(Q, Scalar(25.0f));

	//Process  covariance Matrix P
	P = Mat::eye(Size(mat_size, mat_size), CV_32F);
	setIdentity(P, Scalar(10e2f));

	//Observation Measurement Matrix H
	H = Mat::zeros(Size(mat_size, Z_size), CV_32F);

	//Measurement covariance matrix R
	R = Mat::eye(Size(Z_size, Z_size), CV_32F);
	setIdentity(R, Scalar(20.0f));
	//Intilize State Xk
	Xk = Mat::zeros(mat_size, 1, CV_32F);



	if (filter_type == 1) {
		A.at<float>(0, 1) = A.at<float>(2, 3) = 1.0f;
		Q.at<float>(1, 1) = Q.at<float>(3, 3) = 10.0f;
		H.at<float>(0, 0) = H.at<float>(1, 2) = 1.0f;

	}
	else if (filter_type == 2) {
		A.at<float>(0, 1) = A.at<float>(1, 2) = A.at<float>(3, 4) = A.at<float>(
				4, 5) = 1.0f;
		A.at<float>(0, 2) = A.at<float>(3, 5) = 0.5f;

		Q.at<float>(1, 1) = Q.at<float>(4, 4) = 10.0f;
		Q.at<float>(2, 2) = Q.at<float>(5, 5) = 1.0f;
		H.at<float>(0, 0) = H.at<float>(1, 3) = 1.0f;


	}
}

void kalmanFilter::initialize_filter() {
	//Intialize Kalman FIlter Object
	unsigned int type = CV_32F;
	//KalmanFilter k;

	//kf = k;
	kf.init(mat_size, Z_size, 0, type);


	//Set the parameter matrices into KalmanFilter Object
	kf.measurementMatrix = H;
	kf.measurementNoiseCov = R;
	kf.processNoiseCov = Q;
	kf.transitionMatrix = A;
	kf.errorCovPre = P;




}

void kalmanFilter::applyKalman(Point center) {
	Zk = Mat::zeros(Z_size, 1, CV_32F);
//	Point predicted_center;
//	Point corrected_center;

	bool blob_flag = false;

	if (center.x != 0 || center.y != 0) {
		blob_flag = true;
	}


	if (blob_flag) {
		if (flag) {
			//initialize Xk for the first time
			Xk.at<float>(0) = center.x;

			//Check the filter type and assign y coordinate value accordingly
		if (filter_type == 1) {
			Xk.at<float>(2) = center.y;
		} else {
			Xk.at<float>(3) = center.y;
		}

			//Set the postState
			kf.statePost = Xk;
			flag = false;

		}

	else {

		//Predict the position
		Xk = kf.predict();

			//Take the current observation
			Zk.at<float>(0) = center.x;
			Zk.at<float>(1) = center.y;

			//Correct the measurement
			kf.correct(Zk);

			//Extract the corrected state from filter state
			Mat X_post = kf.statePost;

			//extract correct positions depending on filter type
			if (filter_type == 1) {
				corrected_center = Point(X_post.at<float>(0),
						X_post.at<float>(2));
			} else {
				corrected_center = Point(X_post.at<float>(0),
						X_post.at<float>(3));
			}

			//put the points into vector
			correctedPoints.push_back(corrected_center);
		}
	} else {
		if (!flag) {
			//Predict the position
			Xk = kf.predict();

			//extract predicted positions depending on filter type
			if (filter_type == 1) {
				predicted_center = Point(Xk.at<float>(0), Xk.at<float>(2));
			} else {
				predicted_center = Point(Xk.at<float>(0), Xk.at<float>(3));
			}
			//put the points into vector
			predictedPoints.push_back(predicted_center);
		}

	}

	// Put into the vector
	measurementPoints.push_back(center);

}
Point kalmanFilter::getCurrentPredictedCenter() {
	return predicted_center;
}
Point kalmanFilter::getCurrentCorrectedCenter() {
	return corrected_center;
}


// Predicted Trajectory
std::vector<Point> kalmanFilter::predicted_trajectory() {
	return predictedPoints;
}

// Measurement Trajectory
std::vector<Point> kalmanFilter::measurement_trajectory() {
	return measurementPoints;
}
// corrected Trajectory
std::vector<Point> kalmanFilter::corrected_trajectory() {
	return correctedPoints;
}
