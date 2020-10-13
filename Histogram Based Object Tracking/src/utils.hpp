/* Applied Video Sequence Analysis - Escuela Politecnica Superior - Universidad Autonoma de Madrid
 *
 *	This source code belongs to the template code for
 *	the assignment LAB 4 "Histogram-based tracking"
 *
 *	Header of utilities for LAB4.
 *	Some of these functions are adapted from OpenSource
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 * Date: April 2020
 */
#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <string> 		// for string class
#include <opencv2/opencv.hpp>
#include<vector>
using namespace cv;
struct cvPatch {
	int ID;
	Mat data;
	int x, y;
	int w, h;
};

inline cvPatch initPatch(int id, Mat data, int x, int y, int w, int h) {
	cvPatch B = { id, data, x, y, w, h };
	return B;
}

std::vector<cv::Rect> readGroundTruthFile(std::string groundtruth_path);
std::vector<float> estimateTrackingPerformance(std::vector<cv::Rect> Bbox_GT, std::vector<cv::Rect> Bbox_est);
void showPatch(std::vector<cvPatch> patch);
std::vector<cvPatch> generateCandidates(Mat frame, int numCand, Point start,
		int stride, int h, int w);
Mat visualizeHist(Mat histMat, int bins);

#endif /* UTILS_HPP_ */
