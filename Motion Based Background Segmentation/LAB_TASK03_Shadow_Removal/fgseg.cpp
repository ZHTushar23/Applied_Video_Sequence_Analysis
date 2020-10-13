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

		cout << "Only color version is available in this file." << endl;

	}

	else{
		
		_frame=Frame;

		cv::Mat diffSPLIT[3], splitFrame[3], splitBkg[3],
				updated_bkg_channel[3], bkgInter[3], bgmaskBLUE,
				bgmaskGREEN, bgmaskRED;
		//Absolute Difference between Background and current Frame
		absdiff(_bkg, Frame, _diff);

		//Splitting the difference image
		split(_diff,diffSPLIT);

		//Extract the Background subtraction mask.
		bgmaskBLUE=diffSPLIT[0]>_threshold;
		bgmaskGREEN=diffSPLIT[1]>_threshold;
		bgmaskRED=diffSPLIT[2]>_threshold;
		bitwise_or(bgmaskBLUE,bgmaskGREEN,_bgsmask);
		bitwise_or(bgmaskRED,_bgsmask,_bgsmask);

		//_bgsmask = _diff > _threshold;
		//Calculate the Foreground and Background Logical Mask
		cv::Mat fgLogicalMask = _bgsmask / 255;
		cv::Mat bgLogicalMask = (255 - _bgsmask) / 255;

		split(_frame, splitFrame);

		split(_bkg, splitBkg);
		//Update the background depending on alpha value
		updated_bkg_channel[0] = _alpha * splitFrame[0]
				+ (1 - _alpha) * splitBkg[0];

		updated_bkg_channel[1] = _alpha * splitFrame[1]
				+ (1 - _alpha) * splitBkg[1];

		updated_bkg_channel[2] = _alpha * splitFrame[2]
				+ (1 - _alpha) * splitBkg[2];

		cv::Mat updated_bkg;




		//Update only those pixel belong to the Background(Selective Update)
		bkgInter[0] = bgLogicalMask.mul(updated_bkg_channel[0])
				+ fgLogicalMask.mul(splitBkg[0]);
		bkgInter[1] = bgLogicalMask.mul(updated_bkg_channel[1])
				+ fgLogicalMask.mul(splitBkg[1]);
		bkgInter[2] = bgLogicalMask.mul(updated_bkg_channel[2])
				+ fgLogicalMask.mul(splitBkg[2]);

		vector<Mat> channels;
		channels.push_back(bkgInter[0]);
		channels.push_back(bkgInter[1]);
		channels.push_back(bkgInter[2]);

		//For blind update
		/*vector<Mat> channels;
		channels.push_back(updated_bkg_channel[0]);
		channels.push_back(updated_bkg_channel[1]);
		channels.push_back(updated_bkg_channel[2]);
		 */
		merge(channels, updated_bkg);
		_bkg = updated_bkg;
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

	cv::Mat _frame_hsv;

	cvtColor(_frame, _frame_hsv, CV_BGR2HSV);


	cv::Mat _bkg_hsv;
	cvtColor(_bkg, _bkg_hsv, CV_BGR2HSV);

	//Separating the color channels
	cv::Mat frameSPLIT[3], bkgSPLIT[3];

	split(_frame_hsv,frameSPLIT);
	split(_bkg_hsv,bkgSPLIT);
	//Convert to Double


	Mat div, DH, condition_1a, condition_1b, condition_1, condition_2,
			condition_3, valueDIFF, saturationDIFF, hueDIFF, B1, F1;
	

	frameSPLIT[2].convertTo(F1, CV_64F);
	bkgSPLIT[2].convertTo(B1, CV_64F);

	divide(F1, B1, valueDIFF);
	

	absdiff(frameSPLIT[1], bkgSPLIT[1], saturationDIFF);
	absdiff(frameSPLIT[0], bkgSPLIT[0], hueDIFF);

	//
	min(hueDIFF, 360 - hueDIFF, DH);


	condition_1a = (valueDIFF >= _a) / 255;

	condition_1b = (valueDIFF <= _beta) / 255;


	bitwise_and(condition_1b, condition_1a, condition_1);


	condition_2 = saturationDIFF <= _s_tau;
	condition_3 = DH <= _h_tau;

	//cout << condition_2;

	//converting condition into mask for matrix operation
	condition_2 = condition_2 / 255;
	condition_3 = condition_3 / 255;

	cv::Mat temp;

	bitwise_and(condition_2, condition_3, temp);


	bitwise_and(condition_1, temp, _shadowmask);
	Mat fgLogicalMask = _bgsmask / 255;

	_shadowmask = _shadowmask.mul(fgLogicalMask);
	_shadowmask = _shadowmask * 255;



	//absdiff(_bgsmask, _bgsmask, _shadowmask);// currently void function mask=0 (should create shadow mask)

	absdiff(_bgsmask, _shadowmask, _fgmask); // eliminates shadows from bgsmask
}

//ADD ADDITIONAL FUNCTIONS HERE

