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

using namespace fgseg;
using namespace cv;
//default constructor
bgs::bgs(double threshold, double weight_threshold, bool rgb, double alpha)
{
	_rgb=rgb;
	_weight_threshold = weight_threshold;
	_threshold=threshold;
	_alpha = alpha;
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
		Frame.copyTo(_bkg);

		for (int i = 0; i < num_gaussians; i++) {
			//Random Initialization of variance
			sigma[i] = Mat::ones(Size(_bkg.cols, _bkg.rows), CV_8UC1);
			randu(sigma[i], Scalar(0), Scalar(20));

			//Weight initialization
			weight[i] = Mat::ones(Size(_bkg.cols, _bkg.rows), CV_32FC1);
			randu(sigma[i], Scalar(0), Scalar(1 / num_gaussians));

			//Mean initialized with first frame
			_mu[i] = Mat::zeros(Size(_bkg.cols, _bkg.rows), CV_8UC1);
			Frame.copyTo(_mu[i]);
			
		}

	}
	else{
		cout << "Color currently not supported" << endl;
		exit(1);
	}
}

void bgs::bkgSubtraction(cv::Mat Frame)
{
	if (!_rgb){
		cvtColor(Frame, Frame, COLOR_BGR2GRAY);
		Frame.copyTo(_frame);
		absdiff(_frame, _bkg, _diff);
		_bgsmask = _diff > _threshold;
		//ADD YOUR CODE HERE
		//...
		

		cv::Mat diff;

		for (int i = 0; i < num_gaussians; i++) {
			cv::subtract(_frame, _mu[i], diff);

			//Update the mean of Gaussians
			_mu[i] = _alpha * _frame + (1 - _alpha) * _mu[i];
			//Update the Variance of the Gaussians
			sigma[i] = _alpha * (diff.mul(diff)) + (1 - _alpha) * sigma[i];
		}

		//Array for storing the weights from 5 different gaussians of same pixel.
		double weight_array[num_gaussians] = { };

		// Sum for comparing with the weight threshold Wth.
		double _sum = 0;



		for (int i = 0; i < _bkg.cols; i++) {
			for (int j = 0; j < _bkg.rows; j++) {
				
				for (int k = 0; k < num_gaussians; k++)
					weight_array[k] = weight[k].at<float>(Point(i, j));
				while (1) {
					//get the distance (index) of Max elements from 1st element
					int _index = std::distance(weight_array,
							std::max_element(weight_array,
									weight_array + num_gaussians));
					
					//Check if the sum is already Greater than Wth.
					_sum = _sum + weight_array[_index];
					//Make the current maximum value zero
					weight_array[_index] = 0;
					
					if (_sum >= _weight_threshold) {

						updateWeights(weight_array, i,
								j, _alpha);
						updateBackgroundandMask(i, j, weight_array);
						break;
					}
				}
				_sum = 0;
			}

		}
	}

	else {
		cout << "Color currently not supported" << endl;
		exit(1);
	}
}

//method to detect and remove shadows in the BGS mask to create FG mask
void bgs::removeShadows() {
	if (!_rgb) {


		absdiff(_bgsmask, _bgsmask, _shadowmask);// currently void function mask=0 (should create shadow mask)
		absdiff(_bgsmask, _shadowmask, _fgmask); // eliminates shadows from bgsmask
		//...
	} else {
		cout << "Color currently not supported" << endl;
		exit(1);
	}
}

//ADD ADDITIONAL FUNCTIONS HERE



void bgs::updateWeights(double weight_array[], int col, int row,
		double _alpha) {
	//sums of weights after update for normalization
	double sum_weights = 0;
	for (int k = 0; k < num_gaussians; k++) {
		if (weight_array[k] == 0) {
			//For the selected Gaussians for background model the weights will be zero in the weight_array.
			weight[k].at<float>(Point(col, row)) = (1 - _alpha)
					* weight[k].at<float>(Point(col, row)) + _alpha;
		} else {
			//This other way around
			weight[k].at<float>(Point(col, row)) = (1 - _alpha)
					* weight[k].at<float>(Point(col, row));
		}
		sum_weights = sum_weights + weight[k].at<float>(Point(col, row));
		
	}
	//Normalize the weights
	for (int k = 0; k < num_gaussians; k++) {
		weight[k].at<float>(Point(col, row)) = weight[k].at<float>(
				Point(col, row)) / sum_weights;
	}
}

void bgs::updateBackgroundandMask(int col, int row, double weight_array[]) {
	_bkg.at<uchar>(Point(col, row)) = 0;

	for (int k = 0; k < num_gaussians; k++) {
		////////////////////////////////////////////////////////////////////////////////////////
		// I HAD TO COOMENT OUT THIS THING THOUGH I BELIEVE THIS IS THE COORECT IMPLEMENTATION
		// BUT WITH IS UPDATE PORTION IS JUST STUCKS.   MAY BE DUE TO MEMORY BECAUSE I CHECKED 
		// THE RESOURCE MONITOR IT WHILE RUNNING.WE HAD TO PUT A THRESHOLDING AT THE VERY END
		// TO GENERATE THE MASK. WHICH MIGHT NOT BE THE EXACT IMPLEMNTATION BUT IT SEEMS TO WORK.
		////////////////////////////////////////////////////////////////////////////////////////

		//Check if the pixel falls into the Gaussian that belongs to Background model and create mask.
		/*if (weight_array[k] == 0) {
			Mat std;
			if (_frame.at<uchar>(Point(row, col))
					< _mu[k].at<uchar>(Point(col, row))
							+ sigma[k].at<uchar>(Point(col, row))
					|| _frame.at<uchar>(Point(row, col))
							> _mu[k].at<uchar>(Point(col, row))
									- sigma[k].at<uchar>(Point(col, row))) {
				_bgsmask.at<uchar>(Point(row, col)) = 255;
		 } */
			//Update the Background
			_bkg.at<uchar>(Point(col, row)) = _bkg.at<uchar>(Point(col, row))
					+ (uchar) ((((float) weight[k].at<float>(Point(col, row)))
							* _mu[k].at<uchar>(Point(col, row))));
		}
	}



