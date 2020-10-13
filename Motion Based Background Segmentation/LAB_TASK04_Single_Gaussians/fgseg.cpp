/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1.0: Background Subtraction - Unix version
 *	fgesg.cpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 */

#include <opencv2/opencv.hpp>
#include "fgseg.hpp"
#include <bits/stdc++.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace fgseg;
using namespace cv;
//default constructor
bgs::bgs(double threshold, bool rgb) // @suppress("Class members should be properly initialized")
{
	_rgb=rgb;
	_threshold=threshold;
}

bgs::bgs(double threshold, double alpha, bool selective_bkg_update, bool rgb) // @suppress("Class members should be properly initialized")
{
	_rgb=rgb;
	_alpha=alpha;
	_selective_bkg_update=selective_bkg_update;
	_threshold=threshold;
	_counterThreshold = 100;

}
bgs::bgs(double threshold, double alpha, bool selective_bkg_update, bool rgb,
		double a, double beta, double s_tau, double h_tau) {
	_rgb = rgb;
	_alpha = alpha;
	_selective_bkg_update = selective_bkg_update;
	_threshold = threshold;
	_counterThreshold = 100;
	_a = a;
	_beta = beta;
	_s_tau = s_tau;
	_h_tau = h_tau;
}





//default destructor
bgs::~bgs(void)
{
}

//method to initialize bkg (first frame - hot start)
void bgs::init_bkg(cv::Mat Frame)
{

	if (!_rgb){

		cvtColor(Frame, Frame, COLOR_BGR2GRAY); // to work with gray even if input is color
		//initialize the background
		Frame.copyTo(_bkg);


		//Scalar tmpMean, tmpStd;
		//meanStdDev(Frame, tmpMean, tmpStd);
		//double D = tmpStd.val[0];

		Frame.copyTo(_mu);
		_mu.convertTo(_mu, CV_32FC1);
		_std = Mat::ones(Size(Frame.cols, Frame.rows), CV_32FC1);
		_std = _std * 20;

		_std.convertTo(_std, CV_32FC1);
		pow(_std, 2, _variance);
		_variance.convertTo(_variance, CV_32FC1);


	}

	else{


	}

}

//method to perform BackGroundSubtraction
void bgs::bkgSubtraction(cv::Mat Frame)
{

	if (!_rgb){

		cvtColor(Frame, _frame, COLOR_BGR2GRAY); // to work with gray even if input is color
		absdiff(_bkg, _frame, _diff);

		//Convert the frame to double/float type for matrix operations
		_frame.convertTo(_frame, CV_32FC1);

		Mat frame_double, diff, diff_square;

		//GLB & GUB gaussians upper bound and gaussian lower bound matrices. this is to check if the pixel are inside the gaussians
		Mat GUB = _mu + (_std * 3);
		Mat GLB = _mu - (_std * 2.1);


		//Check if the pixel are within the Gaussians
		Mat condition_1a = (_frame <= GUB) / 255;
		Mat condition_1b = (_frame >= GLB) / 255;
		bitwise_and(condition_1b, condition_1a, _bgsmask);

		_bgsmask = 1 - _bgsmask;
		_bgsmask = _bgsmask * 255;

		//Update the Mean
		 _mu = _alpha * _frame + (1 - _alpha) * _mu;


		//Update the Variance (Standard Deviation).
		absdiff(_frame, _mu, diff);
		pow(diff, 2, diff_square);
		diff_square.convertTo(diff_square, CV_32FC1);
		_variance = _alpha * diff_square + (1 - _alpha) * (_variance);

		//Store back in standard Deviation matrix
		sqrt(_variance, _std);
		_std.convertTo(_std, CV_32FC1);
		


		_mu.copyTo(_bkg);
		_bkg.convertTo(_bkg, CV_8UC1);
	



	}

	else{



	}
}

//method to detect and remove shadows in the BGS mask to create FG mask
//method to detect and remove shadows in the BGS mask to create FG mask
void bgs::removeShadows()
{
	// init Shadow Mask (currently Shadow Detection not implemented)
	//_bgsmask.copyTo(_shadowmask); // creates the mask (currently with bgs)



	//ADD YOUR CODE HERE
	//...
	//Converting color space from BGR to HSV




	absdiff(_bgsmask, _bgsmask, _shadowmask);// currently void function mask=0 (should create shadow mask)

	absdiff(_bgsmask, _shadowmask, _fgmask); // eliminates shadows from bgsmask
}

//ADD ADDITIONAL FUNCTIONS HERE
