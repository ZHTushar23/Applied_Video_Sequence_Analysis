/* Applied Video Sequence Analysis - Escuela Politecnica Superior - Universidad Autonoma de Madrid
 *
 *	This source code belongs to the template code for
 *	the assignment LAB 4 "Histogram-based tracking"
 *
 *	Implementation of utilities for LAB4.
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 * Date: April 2020
 */
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include <vector>
#include "ShowManyImages.hpp"
#include<math.h>
using namespace cv;
using namespace std;

/**
 * Reads a text file where each row contains comma separated values of
 * corners of groundtruth bounding boxes.
 *
 * The annotations are stored in a text file with the format:
 * Row format is "X1, Y1, X2, Y2, X3, Y3, X4, Y4" where Xi and Yi are
 * the coordinates of corner i of the bounding box in frame N, which
 * corresponds to the N-th row in the text file.
 *
 * Returns a list of cv::Rect with the bounding boxes data.
 *
 * @param ground_truth_path: full path to ground truth text file
 * @return bbox_list: list of ground truth bounding boxes of class Rect
 */
std::vector<Rect> readGroundTruthFile(std::string groundtruth_path)
{
	// variables for reading text file
	ifstream inFile; //file stream
	string bbox_values; //line of file containing all bounding box data
	string bbox_value;  //a single value of bbox_values

	vector<Rect> bbox_list; //output with all read bounding boxes

	// open text file
	inFile.open(groundtruth_path.c_str(),ifstream::in);
	if(!inFile)
		throw runtime_error("Could not open groundtrutfile " + groundtruth_path); //throw error if not possible to read file

	// Read each line of groundtruth file
	while(getline(inFile, bbox_values)){

		stringstream linestream(bbox_values); //convert read line to linestream
		//cout << "-->lineread=" << linestream.str() << endl;

		// Read comma separated values of groundtruth.txt
		vector<int> x_values,y_values; 	//values to be read from line
		int line_ctr = 0;						//control variable to read alternate Xi,Yi
		while(getline(linestream, bbox_value, ',')){

			//read alternate Xi,Yi coordinates
			if(line_ctr%2 == 0)
				x_values.push_back(stoi(bbox_value));
			else
				y_values.push_back(stoi(bbox_value));
			line_ctr++;
		}

		// Get width and height; and minimum X,Y coordinates
		double xmin = *min_element(x_values.begin(), x_values.end()); //x coordinate of the top-left corner
		double ymin = *min_element(y_values.begin(), y_values.end()); //y coordinate of the top-left corner

		if (xmin < 0) xmin=0;
		if (ymin < 0) ymin=0;

		double width = *max_element(x_values.begin(), x_values.end()) - xmin; //width
		double height = *max_element(y_values.begin(), y_values.end()) - ymin;//height

		// Initialize a cv::Rect for a bounding box and store it in a std<vector> list
		bbox_list.push_back(Rect(xmin, ymin, width, height));
		//std::cout << "-->Bbox=" << bbox_list[bbox_list.size()-1] << std::endl;
	}
	inFile.close();

	return bbox_list;
}

/**
 * Compare two lists of bounding boxes to estimate their overlap
 * using the criterion IOU (Intersection Over Union), which ranges
 * from 0 (worst) to 1(best) as described in the following paper:
 * Čehovin, L., Leonardis, A., & Kristan, M. (2016).
 * Visual object tracking performance measures revisited.
 * IEEE Transactions on Image Processing, 25(3), 1261-1274.
 *
 * Returns a list of floats with the IOU for each frame.
 *
 * @param Bbox_GT: list of elements of type cv::Rect describing
 * 				   the groundtruth bounding box of the object for each frame.
 * @param Bbox_est: list of elements of type cv::Rect describing
 * 				   the estimated bounding box of the object for each frame.
 * @return score: list of float values (IOU values) for each frame
 *
 * Comments:
 * 		- The two lists of bounding boxes must be aligned, meaning that
 * 		position 'i' for both lists corresponds to frame 'i'.
 * 		- Only estimated Bboxes are compared, so groundtruth Bbox can be
 * 		a list larger than the list of estimated Bboxes.
 */
std::vector<float> estimateTrackingPerformance(std::vector<cv::Rect> Bbox_GT, std::vector<cv::Rect> Bbox_est)
{
	vector<float> score;

	//For each data, we compute the IOU criteria for all estimations
	for(int f=0;f<(int)Bbox_est.size();f++)
	{
		Rect m_inter = Bbox_GT[f] & Bbox_est[f];//Intersection
		Rect m_union = Bbox_GT[f] | Bbox_est[f];//Union

		score.push_back((float)m_inter.area()/(float)m_union.area());
	}

	return score;
}

//This method is to view patch

void showPatch(std::vector<cvPatch> trajectoryPoints) {

	///DRAW PREDICTED TREJECTORY
	cvPatch p1;
	vector<cvPatch>::iterator ic;
	for (ic = trajectoryPoints.begin(); ic != trajectoryPoints.end(); ic++) {

		p1 = *ic;

		imshow("Patch", p1.data);
		//ShowManyImages("patch", 1, p1.data);

	}


}

std::vector<cvPatch> generateCandidates(Mat frame, int numCand, Point start,
		int stride, int h, int w) {

	//Calculate the range
	int range = sqrt(numCand);

	int id = 0;
	vector<cvPatch> patches;

	//Calculate the Top left position of the search area
	int upperX = start.x - (stride * (range / 2));
	int upperY = start.y + (stride * (range / 2));

	//Calculate the bottom right position of the search area
	int lowerX = start.x + (stride * (range / 2));
	int lowerY = start.y - (stride * (range / 2));


//Go through the search Area and extract patch
	for (int i = upperX; i <= lowerX; i = i + stride) {
		for (int j = upperY; j >= lowerY; j = j - stride) {

			if (i < frame.cols && j < frame.rows && i > 0 && j > 0) {

				int rectH =
						j + h >= (int) frame.rows ?
								h - (j + h - frame.rows) : h;
				int rectW =
						i + w >= (int) frame.cols ?
								w - (i + w - frame.cols) : w;
				Mat patch = Mat(frame(Rect(i, j, rectW, rectH)));
			//Construct a cvPatch structure which contains data(matrix),id ,x and y position and height and weight of candidate patch.
				cvPatch cp = initPatch(id, patch, i, j, rectW, rectH);
			//Push back into a vector
				patches.push_back(cp);
			}
		}
	}

	return patches;
}




//Method for visualizing the Histogram.
Mat visualizeHist(Mat histMat, int bins) {
	histMat = histMat * 256;
	int histSize = bins;
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double) hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	for (int i = 1; i < histSize; i++) {
		line(histImage,
				Point(bin_w * (i - 1),
						hist_h - cvRound(histMat.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(histMat.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);
	}
	/// Display
//	namedWindow("Histogram", CV_WINDOW_AUTOSIZE);
//	imshow("Histogram", histImage);
	return histImage;
}


