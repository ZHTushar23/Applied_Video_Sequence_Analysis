/*
 * fuseModule.cpp
 *
 *  Created on: May 8, 2020
 *      Author: mohammad
 */


#include <opencv2/opencv.hpp>
#include <vector>
#include "gradientModule.hpp"
#include "opencv2/imgcodecs.hpp"
#include <math.h>
#include "ShowManyImages.hpp"
using namespace cv;
using namespace std;
using namespace grad;

cvPatch gradientModule::track(Mat model, vector<cvPatch> patches, int bins) {

	//Compute the gradient for patches.
	vector<cvPatch> grad_candlist = computePatchGradient(patches, bins);
	Mat gradModel = calculateGradient(model, bins);
	//Calculate the L2 distance and get best candidate
	cvPatch result_patch = compareGradients(grad_candlist, gradModel);

	return result_patch;
}


//This method compares patches by calculating l2 distance
cvPatch gradientModule::compareGradients(vector<cvPatch> grad_candlist,
		Mat gradModel) {
	//calculate the distance
	vector<cvPatch>::iterator pd;
	vector<double> scores;
	int index;

	cvPatch patch_holder;
	//iterate over the gradient patches
	for (pd = grad_candlist.begin(); pd != grad_candlist.end(); pd++) {
		patch_holder = *pd;
		//Compute the l2 distance between the model and candidates
		double dist = norm(gradModel, patch_holder.data, NORM_L2);
		//push into vector
		scores.push_back(dist);
	}
	//Get the index of minimum score.
	index = std::min_element(scores.begin(), scores.end()) - scores.begin();

	return grad_candlist[index];
}


std::vector<cvPatch> gradientModule::computePatchGradient(
		vector<cvPatch> patches,
		int bins) {

	vector<cvPatch> grad_candlist;

	//Iterate over the patches and calculate gradient for them.
	vector<cvPatch>::iterator pt;
	for (pt = patches.begin(); pt != patches.end(); pt++) {
		cvPatch holder;
		Mat temp, grad;
		holder = *pt;
		temp = holder.data;
		//compute gradient featurs of each candidate patch
		grad = calculateGradient(temp, bins);
		grad_candlist.push_back(
				initPatch(holder.ID, grad, holder.x, holder.y, holder.w,
						holder.h));
	}
	return grad_candlist;
}

//This method Computes HoG
Mat gradientModule::calculateGradient(Mat model, int bin) {
	//HOGDescriptor hog;
	vector<float> descriptors;
	vector<Point> loc;
	Mat gray;

	//convert color space
	cvtColor(model, gray, CV_BGR2GRAY);
	gray.convertTo(gray, CV_8UC1);
	//resize the image to match with Hog Descriptor size
	cv::resize(gray, gray, cv::Size(64, 128));
	//Compute Gradient features.
	HOGDescriptor *hog = new HOGDescriptor();
	hog->nbins = bin;
	hog->compute(gray, descriptors, Size(2, 2), Size(0, 0), loc);

	//Convert the vector into Matrix
	Mat feat = Mat(descriptors).clone();

	return feat;

}

