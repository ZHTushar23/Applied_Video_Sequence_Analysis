/*
 * fuseModule.cpp
 *
 *  Created on: May 8, 2020
 *      Author: mohammad
 */
#include <opencv2/opencv.hpp>

#include "gradientModule.hpp"
#include "fuseModule.hpp"
#include "colorModule.hpp"
#include "opencv2/imgcodecs.hpp"
#include <math.h>
#include "ShowManyImages.hpp"
#include <vector>

using namespace fuse;

colo::colorModule colorMod;
grad::gradientModule gradMod;

cvPatch fuseModule::track(Mat model, vector<cvPatch> patches, int bins,
		int nbin,
		int mode) {

	vector<cvPatch> hist_candlist, grad_candlist;
	cvPatch result_patch;

	//Matrices for histogram
	Mat b_histModel, g_histModel, r_histModel, gray_histModel, h_histModel,
			s_histModel, grad;
	Mat histList[] = { b_histModel, g_histModel, r_histModel, h_histModel,
			s_histModel, gray_histModel };

// Computes the color features using colorModlue Object
//Computes Gradient features using gradientModule Object
//depending on the mode selection we can set gradient+color channel R/G/B/H/S/Gray
	if (mode == 1) {
		histList[0] = colorMod.calculateChannelB(model, bins);
		grad = gradMod.calculateGradient(model, nbin);
	}
	if (mode == 2) {
		histList[1] = colorMod.calculateChannelG(model, bins);
		grad = gradMod.calculateGradient(model, nbin);
	}
	if (mode == 3) {
		histList[2] = colorMod.calculateChannelR(model, bins);
		grad = gradMod.calculateGradient(model, nbin);
	}

	if (mode == 4) {
		histList[3] = colorMod.calculateChannelH(model, bins);
		grad = gradMod.calculateGradient(model, nbin);
	}
	if (mode == 5) {
		histList[4] = colorMod.calculateChannelS(model, bins);
		grad = gradMod.calculateGradient(model, nbin);
	}
	if (mode == 6) {
		histList[5] = colorMod.calculateChannelGray(model, bins);
		grad = gradMod.calculateGradient(model, nbin);
	}

	//Compute the Histogram of all candidates
	hist_candlist = colorMod.computePatchHist(patches, bins, mode);

	//Compute the Gradient of all candidates
	grad_candlist = gradMod.computePatchGradient(patches, nbin);

	//find the best candidate according to both features and fuse the scores then return the best candidate
	result_patch = computeFusedScore(hist_candlist, grad_candlist,
			histList[mode - 1], grad);

	return result_patch;
}


cvPatch fuseModule::computeFusedScore(vector<cvPatch> hist_candlist,
		vector<cvPatch> grad_candlist, Mat histModel, Mat gradModel) {

	//calculate the distance
	vector<cvPatch>::iterator pd;
	vector<double> scoresH, scoresG, scoresF;
	int index;

	cvPatch patch_holder_grad;

	//iterate over each candidate gradient and compute L2 distance between candidate and model
	for (pd = grad_candlist.begin(); pd != grad_candlist.end(); pd++) {
		patch_holder_grad = *pd;
		//Calculate L2 distance
		double dist = norm(patch_holder_grad.data, gradModel, NORM_L2);
		scoresG.push_back(dist);
		
	}

	//Normalize the L2 distance calculated for gradient module
	int max = std::max_element(scoresG.begin(), scoresG.end())
			- scoresG.begin();
	int min = std::min_element(scoresG.begin(), scoresG.end())
			- scoresG.begin();
	for (int i = 0; i < int(scoresG.size()); i++) {
		scoresG[i] = (scoresG[i] - scoresG[min])
				/ (scoresG[max] - scoresG[min]);

	}

		cvPatch patch_holder_hist;

		//iterate over each candidate histogram and compute Bhattcharyya distance between candidate and model
	for (pd = hist_candlist.begin(); pd != hist_candlist.end(); pd++) {
			patch_holder_hist = *pd;
			//calculate bhattacharyya distance and push into vector.
		scoresH.push_back(
					compareHist(patch_holder_hist.data, histModel,
						CV_COMP_BHATTACHARYYA));
	}
	//Normalize the Bhattacharyya distance calculated for gradient module
	int max2 = std::max_element(scoresH.begin(), scoresH.end())
			- scoresH.begin();
	int min2 = std::min_element(scoresH.begin(), scoresH.end())
			- scoresH.begin();
	for (int i = 0; i < int(scoresH.size()); i++) {
		scoresH[i] = (scoresH[i] - scoresH[min2])
				/ (scoresH[max2] - scoresH[min2]);
	}

	//just to initialize the vector.
	scoresF = scoresH;

		//Add the scores from both module
	for (int i = 0; i < int(scoresH.size()); i++) {
		scoresF[i] = scoresG[i] + scoresH[i];

	}
	
	//Get the index of minimum score.
	index = std::min_element(scoresF.begin(), scoresF.end()) - scoresF.begin();

	return hist_candlist[index];
}

