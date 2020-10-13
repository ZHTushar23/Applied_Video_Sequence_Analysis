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
	_counterThreshold = 200;
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
		//Counter Matrix for Counting pixel Occurrence (Lab 1.1.3)
		counter = Mat::zeros(Size(Frame.cols, Frame.rows), CV_8UC1);
	}

	else{

		Frame.copyTo(_bkg);
	}

}

//method to perform BackGroundSubtraction
void bgs::bkgSubtraction(cv::Mat Frame)
{

	if (!_rgb){

		cvtColor(Frame, _frame, COLOR_BGR2GRAY); // to work with gray even if input is color

		//Absolute Difference between Background and current Frame
		absdiff(_bkg, _frame, _diff);

		//Extract the Background subtraction mask.
		_bgsmask = _diff > _threshold;

		//Calculate the Foreground and Background Logical Mask
		cv::Mat fgLogicalMask = _bgsmask / 255;
		cv::Mat bgLogicalMask = (255 - _bgsmask) / 255;

		if (_selective_bkg_update) {


			//Reset the counter for pixel  belongs to the background
			counter = counter.mul(fgLogicalMask);
			//Increment counter for the pixel that belongs foreground
			counter = counter + fgLogicalMask;


			// logical mask for those pixels those exceeded the threshold.
			cv::Mat counter_logical_mask = (counter > _counterThreshold) / 255;

			//update the current frame for those pixel whose counter exceed the threshold value
			cv::Mat updated_frame = counter_logical_mask.mul(_frame);

			//calculate the invert mask of counter_logical_mask
			cv::Mat inv_counter_logical_mask = (1 - counter_logical_mask);

			//store the value of pixel whose counter didn't exceed the counter threshold
			cv::Mat updatedBg = _bkg.mul(inv_counter_logical_mask);

			//Update the background for removing stationary object
			_bkg = updated_frame + updatedBg;


			//Update the background depending on alpha value
			cv::Mat updated_bkg = _alpha * _frame + (1 - _alpha) * _bkg;

			//Update only those pixel belong to the Background(Selective Update)
			_bkg = bgLogicalMask.mul(updated_bkg) + fgLogicalMask.mul(_bkg);
		}
		
		else {
			//Update the background depending on alpha value
			cv::Mat updated_bkg = _alpha * _frame + (1 - _alpha) * _bkg;

			//Blind Update
			_bkg = updated_bkg;
		}

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

