/* Applied Video Sequence Analysis - Escuela Politecnica Superior - Universidad Autonoma de Madrid
 *
 *	This source code belongs to the template (sample program)
 *	provided for the assignment LAB 4 "Histogram-based tracking"
 *
 *	This code has been tested using:
 *	- Operative System: Ubuntu 18.04
 *	- OpenCV version: 3.4.4
 *	- Eclipse version: 2019-12
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 * Date: April 2020
 */
//includes
#include <stdio.h> 								//Standard I/O library
#include <vector>
#include <numeric>								//For std::accumulate function
#include <string> 								//For std::to_string function
#include <opencv2/opencv.hpp>					//opencv libraries
#include "utils.hpp"							//for functions readGroundTruthFile & estimateTrackingPerformance
#include "colorModule.hpp"
#include "gradientModule.hpp"
#include "fuseModule.hpp"
//namespaces
using namespace cv;
using namespace std;
using namespace colo;
//main function
int main(int argc, char ** argv)
{
	//PLEASE CHANGE 'dataset_path' & 'output_path' ACCORDING TO YOUR PROJECT
	std::string dataset_path =
			"/home/mohammad/eclipse-workspace/Lab 4.0/datasets";//dataset location.
	std::string output_path =
			"/home/mohammad/eclipse-workspace/Lab 4.0/outvideos";//location to save output videos

	// dataset paths
	std::string sequences[] = { "/bolt1" }; //test data for lab4.1, 4.3 & 4.5
//							    "/car1", "/ball", "/bag"							//test data for lab4.2
//							   "ball2","basketball",						//test data for lab4.4
//							   "bag","ball","road",};						//test data for lab4.6
	std::string image_path = "%08d.jpg"; 									//format of frames. DO NOT CHANGE
	std::string groundtruth_file = "groundtruth.txt"; 						//file for ground truth data. DO NOT CHANGE
	int NumSeq = sizeof(sequences)/sizeof(sequences[0]);					//number of sequences
	Mat frame;

	////////////////////////////////////////////
	// Create necessary objects and variables//
	///////////////////////////////////////////
	vector<cvPatch> candidate_list;
	colo::colorModule cm;
	grad::gradientModule gm;
	fuse::fuseModule fm;

	Rect result_rect;


	////////////////////////////////////////////////////
	//Loop for all sequence of each category
	for (int s=0; s<NumSeq; s++ )
	{
		//current Frame
		int frame_idx=0;								//index of current Frame
		std::vector<Rect> list_bbox_est, list_bbox_gt;	//estimated & groundtruth bounding boxes
		std::vector<double> procTimes;					//vector to accumulate processing times

		std::string inputvideo = dataset_path + "/" + sequences[s] + "/img/" + image_path; //path of videofile. DO NOT CHANGE
		VideoCapture cap(inputvideo);	// reader to grab frames from videofile

		//check if videofile exists
		if (!cap.isOpened())
			throw std::runtime_error("Could not open video file " + inputvideo); //error if not possible to read videofile

		// Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
		cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH),cap.get(cv::CAP_PROP_FRAME_HEIGHT));//cv::Size frame_size(700,460);
		VideoWriter outputvideo(output_path+"outvid_" + sequences[s]+".avi",CV_FOURCC('X','V','I','D'),10, frame_size);	//xvid compression (cannot be changed in OpenCV)

		//Read ground truth file and store bounding boxes
		std::string inputGroundtruth = dataset_path + "/" + sequences[s] + "/" + groundtruth_file;//path of groundtruth file. DO NOT CHANGE
		list_bbox_gt = readGroundTruthFile(inputGroundtruth); //read groundtruth bounding boxes

		//main loop for the sequence
		std::cout << "Displaying sequence at " << inputvideo << std::endl;
		std::cout << "  with groundtruth at " << inputGroundtruth << std::endl;
		Mat model;
		bool flag = false;
		for (;;) {
			//get frame & check if we achieved the end of the videofile (e.g. frame.data is empty)
			cap >> frame;
			if (!frame.data)
				break;

			//Time measurement
			double t = (double)getTickCount();
			frame_idx=cap.get(cv::CAP_PROP_POS_FRAMES);			//get the current frame

			////////////////////////////////////////////////////////////////////////////////////////////
			//DO TRACKING
			//Change the following line with your own code


			//Take the rectangle from the 1st frame a model.


			Mat aux;
			frame.copyTo(aux);

			//////////////////////////////////////////////////////////////////////////
			//////////////////////Parameter To Adjust////////////////////////////////
			/////////////////////////////////////////////////////////////////////////

			//Change between different features 1=Color Feature ,2=Gradient Feature 3=Fusion of color and gradient.
			int feature = 1;

			//Number of candidate
			int canNum = 81;

			//Compute the histogram for different channel mode( 1,2,3,4,5,6=B,G,R,H,S,Gray).(Only for color module)
			int mode = 4;

			//Stride of the candidate extraction windows
			int stride = 1;

			//Number of histogram bin (Only for Color module)
			int binNum = 16;

			//Number of Bin for HoG.(Only for Gradient Module)
			int nbin = 9;

			//Height of the rectangle
			int height = 60;
			//width of the rectangle
			int width = 30;

			///////////////////////////////////////////////////////////////////////
			///////////////////////////////////////////////////////////////////////
			//for the 1st time pass the position of the initial model to extract candidates

			if (!flag) {
				Mat temp;
				frame.copyTo(temp);
				model = Mat(
						temp(
								Rect(list_bbox_gt[0].x, list_bbox_gt[0].y,
										list_bbox_gt[0].width,
										list_bbox_gt[0].height)));

				//Extract the coordinate of the 1st frame.
				Point start = Point(list_bbox_gt[0].x, list_bbox_gt[0].y);
				candidate_list = generateCandidates(aux, canNum, start, stride,
						height, width);
				flag = true;

				//from 2nd frame pass the position of that has been detected in the previous frame  to extract candidates
			} else {
				candidate_list = generateCandidates(aux, canNum,
						Point(result_rect.x, result_rect.y), stride, height,
						width);
			}

			cvPatch result_patch;
			//Select different feature
			if (feature == 1) {
				//call the track method of color module
				result_patch = cm.track(model, candidate_list, binNum, mode);
			} else if (feature == 2) {
				//call the track method of gradient module
				result_patch = gm.track(model, candidate_list, nbin);
			} else if (feature == 3) {
				//call the track method of fused module
				result_patch = fm.track(model, candidate_list, binNum, nbin,
						mode);
			}

			//Construct the the estimated rectangle
			result_rect = Rect(result_patch.x, result_patch.y, result_patch.w,
					result_patch.h);
			
			list_bbox_est.push_back(result_rect);//we use a fixed value only for this demo program. Remove this line when you use your code

			
		
			////////////////////////////////////////////////////////////////////////////////////////////

			//Time measurement
			procTimes.push_back(
					((double) getTickCount() - t) * 1000.
							/ cv::getTickFrequency());
			//std::cout << " processing time=" << procTimes[procTimes.size()-1] << " ms" << std::endl;



			// plot frame number & groundtruth bounding box for each frame
			putText(frame, std::to_string(frame_idx), cv::Point(10, 15),
					FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255)); //text in red
			rectangle(frame, list_bbox_gt[frame_idx - 1], Scalar(0, 255, 0));//draw bounding box for groundtruth
			rectangle(frame, list_bbox_est[frame_idx - 1], Scalar(0, 0, 255)); //draw bounding box (estimation)


			outputvideo.write(frame);	//save frame to output video
			//imshow("MOdel", model);
			ShowManyImages(
					"1.Frame||2.Target Model||3.Histogram of Best candidate", 3,
					frame, model, visualizeHist(result_patch.data, binNum));
			//show & save data
			imshow(
					"Tracking for " + sequences[s]
							+ " (Green=GT, Red=Estimation)", frame);
			//exit if ESC key is pressed
			if(waitKey(30) == 27) break;
		}

		//comparison groundtruth & estimation
		vector<float> trackPerf = estimateTrackingPerformance(list_bbox_gt, list_bbox_est);

		//print stats about processing time and tracking performance
		std::cout << "  Average processing time = " << std::accumulate( procTimes.begin(), procTimes.end(), 0.0) / procTimes.size() << " ms/frame" << std::endl;
		std::cout << "  Average tracking performance = " << std::accumulate( trackPerf.begin(), trackPerf.end(), 0.0) / trackPerf.size() << std::endl;



		//release all resources
		cap.release();			// close inputvideo
		outputvideo.release(); 	// close outputvideo
		destroyAllWindows(); 	// close all the windows
	}
	printf("Finished program.");
	return 0;
}
