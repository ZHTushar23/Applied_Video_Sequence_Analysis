/*
 * colorModule.cpp
 *
 *  Created on: Apr 29, 2020
 *      Author: Mohammad
 */

#include <opencv2/opencv.hpp>
#include <vector>
#include "colorModule.hpp"
#include "gradientModule.hpp"
#include "opencv2/imgcodecs.hpp"
#include <math.h>
#include "ShowManyImages.hpp"

using namespace cv;
using namespace std;
using namespace colo;
grad::gradientModule gradmod;


cvPatch colorModule::track(Mat model, vector<cvPatch> patches,
		int bins,
		int mode) {

	//Vector for candidate histogram.
	vector<cvPatch> hist_candlist;

	cvPatch result_patch;

	//Matrices for histogram
	Mat b_histModel, g_histModel, r_histModel, gray_histModel, h_histModel,
			s_histModel;
	Mat histList[] = { b_histModel, g_histModel, r_histModel, h_histModel,
			s_histModel, gray_histModel };

	//Calculate the histogram for Different color channels depending on mode selection for Model patch
	if (mode == 1) {
		histList[0] = calculateChannelB(model, bins);
	}
	if (mode == 2) {
		histList[1] = calculateChannelG(model, bins);
	}
	if (mode == 3) {
		histList[2] = calculateChannelR(model, bins);
	}
	if (mode == 4) {
		histList[3] = calculateChannelH(model, bins);
	}
	if (mode == 5) {
		histList[4] = calculateChannelS(model, bins);
	}
	if (mode == 6) {
		histList[5] = calculateChannelGray(model, bins);

	}


	//Compute the histogram for all candidate patches
	hist_candlist = computePatchHist(patches, bins, mode);

	//Calculate the battacharyya distance and get best candidate
	result_patch = computeDistance(hist_candlist, histList[mode - 1]);


	return result_patch;
}

//this method computes distance between model and patch
cvPatch colorModule::computeDistance(vector<cvPatch> hist_candlist,
		Mat histModel) {
	//Define necessary variables
	vector<cvPatch>::iterator pd;
	vector<double> scores;
	int index;
	cvPatch patch_holder;

	//iterate over all the candidate histogram and compare them with our model
	//then calculate bhattacharya distance and return the patch with least distance with model.
	for (pd = hist_candlist.begin(); pd != hist_candlist.end(); pd++) {
		patch_holder = *pd;

		//Calculate score
		double score = compareHist(patch_holder.data, histModel,
				CV_COMP_BHATTACHARYYA);

		//push into scores vector
		scores.push_back(score);
	}

	//Get the index of minimum scores vector.
	index = std::min_element(scores.begin(), scores.end()) - scores.begin();

	//Return the patch that has the minimum distance with model
	return hist_candlist[index];
}




//This method Computes the histogram for all the candidate patches
std::vector<cvPatch> colorModule::computePatchHist(vector<cvPatch> patches,
		int bins, int mode) {
	//Define necessary variables
	Mat b_hist, g_hist, r_hist, h_hist, s_hist, gray_hist;
	vector<cvPatch> hist_candlist;
	cvPatch holder;
	Mat temp, temp_histH, temp_histS;

	
	//Iterate over the patches and calculate histogram for them.
	vector<cvPatch>::iterator pt;
	for (pt = patches.begin(); pt != patches.end(); pt++) {
		holder = *pt;
		temp = holder.data;

		//Select the the appropriate  color channel depending on mode selection
		if (mode == 4) {
			//Compute histogram and push it to candidate histogram vector
			temp_histH = calculateChannelH(temp, bins);
			hist_candlist.push_back(
					initPatch(holder.ID, temp_histH, holder.x, holder.y,
							holder.w, holder.h));
		}

		else if (mode == 5) {
			//Compute histogram and push it to candidate histogram vector
			temp_histS = calculateChannelS(temp, bins);
			hist_candlist.push_back(
					initPatch(holder.ID, temp_histS, holder.x, holder.y,
							holder.w, holder.h));
		}

		else if (mode == 3) {
			//Compute histogram and push it to candidate histogram vector
			r_hist = calculateChannelR(temp, bins);
			hist_candlist.push_back(
					initPatch(holder.ID, r_hist, holder.x, holder.y, holder.w,
							holder.h));
		}

		else if (mode == 2) {
			//Compute histogram and push it to candidate histogram vector
			g_hist = calculateChannelG(temp, bins);
			hist_candlist.push_back(
					initPatch(holder.ID, g_hist, holder.x, holder.y, holder.w,
							holder.h));
		}

		else if (mode == 1) {
			//Compute histogram and push it to candidate histogram vector
			b_hist = calculateChannelB(temp, bins);
			hist_candlist.push_back(
					initPatch(holder.ID, b_hist, holder.x, holder.y, holder.w,
							holder.h));
		}

		else if (mode == 6) {
			//Compute histogram and push it to candidate histogram vector
			gray_hist = calculateChannelGray(temp, bins);
			hist_candlist.push_back(
					initPatch(holder.ID, gray_hist, holder.x, holder.y,
							holder.w, holder.h));
		}

		else {
			cout << "Please enter a valid choice" << endl;
		}

	}
	return hist_candlist;
}

//This method computes histogram of G channel
Mat colorModule::calculateChannelG(Mat model, int bin) {
	/// Separate the image in 3 places ( B, G and R )
	Mat bgr_planes[3];
	split(model, bgr_planes);


		/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float *histRange = { range };

	bool uniform = true;
	bool accumulate = false;

	Mat g_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &bin, &histRange, uniform,
			accumulate);


		/// Normalize the result to [ 0,1 ]
	normalize(g_hist, g_hist, 0, 1, NORM_MINMAX, -1, Mat());


	return g_hist;

}
//This method computes histogram of B channel
Mat colorModule::calculateChannelB(Mat model, int bin) {
	/// Separate the image in 3 places ( B, G and R )
	Mat bgr_planes[3];
	split(model, bgr_planes);

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float *histRange = { range };

	bool uniform = true;
	bool accumulate = false;

	Mat b_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &bin, &histRange, uniform,
			accumulate);

	/// Normalize the result to [ 0,1 ]
	normalize(b_hist, b_hist, 0, 1, NORM_MINMAX, -1, Mat());

	return b_hist;

}
//This method computes histogram of R channel
Mat colorModule::calculateChannelR(Mat model, int bin) {
	/// Separate the image in 3 places ( B, G and R )
	Mat bgr_planes[3];
	split(model, bgr_planes);


	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float *histRange = { range };

	bool uniform = true;
	bool accumulate = false;

	Mat r_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &bin, &histRange,
			uniform, accumulate);


	/// Normalize the result to [ 0, 1 ]
	normalize(r_hist, r_hist, 0, 1, NORM_MINMAX, -1, Mat());



	return r_hist;

}
//This method computes histogram of S channel
Mat colorModule::calculateChannelS(Mat model, int bin) {
	/// Separate the image in 3 places ( B, G and R )
	Mat hsv, hsv_planes[3];
	cvtColor(model, hsv, cv::COLOR_BGR2HSV);

	split(hsv, hsv_planes);

	//Set range for rgb color space
	// hue varies from 0 to 179, saturation from 0 to 255
	float s_ranges[] = { 0, 256 };
	const float *rangesS = { s_ranges };

	bool uniform = true;
	bool accumulate = false;

	Mat s_hist;

	/// Compute the histograms:
	calcHist(&hsv_planes[1], 1, 0, Mat(), s_hist, 1, &bin, &rangesS, uniform,
			accumulate);


	/// Normalize the result to [ 0, 1 ]
	normalize(s_hist, s_hist, 0, 1, NORM_MINMAX, -1, Mat());

	return s_hist;

}
//This method computes histogram of H channel
Mat colorModule::calculateChannelH(Mat model, int bin) {
	/// Separate the image in 3 places ( B, G and R )
	Mat hsv, hsv_planes[3];
	cvtColor(model, hsv, cv::COLOR_BGR2HSV);

	split(hsv, hsv_planes);

	//Set range for rgb color space
	float s_ranges[] = { 0, 256 };
	const float *rangesS = { s_ranges };

	bool uniform = true;
	bool accumulate = false;

	Mat h_hist;

	/// Compute the histograms:
	calcHist(&hsv_planes[0], 1, 0, Mat(), h_hist, 1, &bin, &rangesS, uniform,
			accumulate);


		/// Normalize the result to [ 0, 1 ]
	normalize(h_hist, h_hist, 0, 1, NORM_MINMAX, -1, Mat());


	return h_hist;

}
//This method computes histogram of Gray channel
Mat colorModule::calculateChannelGray(Mat model, int bin) {
	/// Separate the image in 3 places ( B, G and R )
	Mat gray, gray_planes;

	cvtColor(model, gray, cv::COLOR_BGR2GRAY);

	//split(gray, gray_planes);

	//Set range for gray space
	float s_ranges[] = { 0, 256 };
	const float *rangesS = { s_ranges };

	bool uniform = true;
	bool accumulate = false;

	Mat gray_hist;

	/// Compute the histograms:
	calcHist(&gray, 1, 0, Mat(), gray_hist, 1, &bin, &rangesS, uniform,
			accumulate);

	/// Normalize the result to [ 0, 1 ]
	normalize(gray_hist, gray_hist, 0, 1, NORM_MINMAX, -1, Mat());

	return gray_hist;

}

