/* Applied Video Analysis of Sequences (AVSA)
 *
 *	LAB3: Single Object Tracking
 *
 */

//system libraries C/C++
#include <stdio.h>
#include <iostream>
#include <sstream>
#include<vector>

#include <unistd.h>
//opencv libraries
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

//Header ShowManyImages
#include "ShowManyImages.hpp"

//include for blob-related functions
#include "blobs.hpp"

//Include Kalman Filter Class

#include "kalmanFilter.hpp"

//namespaces
using namespace cv; //avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)
using namespace std;

#define MIN_WIDTH 50
#define MIN_HEIGHT 50




//main function
int main(int argc, char **argv)
 {

	//Select Filter Type (1. Constant Velocity 2. Constant Acceleration )
	///////////////////////////
	int filterType = 1;
	/////////////////////////



	Mat frame; // current Frame
	Mat fgmask; // foreground mask
	std::vector<cvBlob> bloblist; // list for blobs
	std::vector<cvBlob> bloblistFiltered; // list for blobs
	
	// STATIONARY BLOBS
	Mat fgmask_history; // STATIONARY foreground mask
	Mat sfgmask; // STATIONARY foreground mask
	std::vector<cvBlob> sbloblist; // list for STATIONARY blobs
	std::vector<cvBlob> sbloblistFiltered; // list for STATIONARY blobs



	double t, acum_t; //variables for execution time
	int t_freq = getTickFrequency();

	string dataset_path =
			"/home/mohammad/eclipse-workspace/Lab.03_AVS2020_task_01/AVSA_Lab3_datasets/dataset_lab3/lab3.2"; //SET THIS DIRECTORY according to your download
	string dataset_cat[1] = { "" };



	string baseline_seq[4] = { "video2.mp4", "video3.mp4", "video5.mp4",
			"video6.mp4" };

	string image_path = ""; //path to images - this format allows to read consecutive images with filename inXXXXXX.jpq (six digits) starting with 000001






		int NumCat = sizeof(dataset_cat) / sizeof(dataset_cat[0]); //number of categories (have faith ... it works! ;) ... each string size is 32 -at leat for the current values-)

	//Loop for all categories
	for (int c = 0; c < NumCat; c++) {


		int NumSeq = sizeof(baseline_seq) / sizeof(baseline_seq[0]); //number of sequences per category ((have faith ... it works! ;) ... each string size is 32 -at leat for the current values-)

		//Loop for all sequence of each category
		for (int s = 0; s < NumSeq; s++) {
			VideoCapture cap;  //reader to grab videoframes

			//Compose full path of images
			string inputvideo = dataset_path + "/" + dataset_cat[c] + "/"
					+ baseline_seq[s] + image_path;
			cout << "Accessing sequence at " << inputvideo << endl;

			//open the video file to check if it exists
			cap.open(inputvideo);
			if (!cap.isOpened()) {
				cout << "Could not open video file " << inputvideo << endl;
				return -1;
			}



			//Initialize MOG Model
			Ptr<BackgroundSubtractorMOG2> pMOG2 =
					cv::createBackgroundSubtractorMOG2();
			pMOG2->setVarThreshold(16);
			pMOG2->setHistory(100);
//
			//main loop
			Mat img; // current Frame

			//Initialize Kalman Filter
			kl::kalmanFilter kalman(filterType);
			kalman.initialize_matrices();
			kalman.initialize_filter();


			int it = 1;
			acum_t = 0;

			for (;;) {

				//get frame
				cap >> img;

				//check if we achieved the end of the file (e.g. img.data is empty)
				if (!img.data)
					break;


				//Time measurement
				t = (double) getTickCount();

				//copy frame
				img.copyTo(frame);

				//Apply Background subtraction model
				double learningrate = 0.0001;
				pMOG2->apply(frame, fgmask, learningrate);

				//Apply threshold for remove shadow
				threshold(fgmask, fgmask, 128, 255, THRESH_BINARY);



				//Apply Morphological Operation
				applyMorphologicalOp(fgmask);

				
				// Extract the blobs in fgmask
				int connectivity = 4;
				extractBlobs(fgmask, bloblist, connectivity);


				//Remove Small blobs
				removeSmallBlobs(bloblist, bloblistFiltered, MIN_WIDTH,
						MIN_HEIGHT);

				//Get the coordinate of the blob
				int centerX = 0, centerY = 0;
				extractCenterOfLargestBlob(bloblistFiltered, centerX, centerY);



				//Apply Kalman Filter
				Point center = Point(centerX, centerY);
				kalman.applyKalman(center);




				//Time measurement
				t = (double) getTickCount() - t;
				acum_t = +t;


				//Draw Predicted , Corrected and Measurement Trajectory
				drawTrejectory(frame, kalman.predicted_trajectory(),
						Scalar(255, 0, 0), 3, false);

				drawTrejectory(frame, kalman.corrected_trajectory(),
						Scalar(0, 0, 255), 3, true);

				drawTrejectory(frame, kalman.measurement_trajectory(),
						Scalar(0, 255, 0), 2.5, false);


				//Put the Text in The Frame
				putText(frame, "Measurement", Point(10, 30),
						FONT_HERSHEY_SIMPLEX, 0.8, Scalar(10, 255, 0), 2);
				putText(frame, "Corrected", Point(10, 70), FONT_HERSHEY_SIMPLEX,
						0.8, Scalar(0, 0, 255), 2);
				putText(frame, "Predicted", Point(10, 110),
						FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);

				//Show the result

				ShowManyImages("Tracking", 3, fgmask, frame,
						paintBlobImage(frame, bloblistFiltered, true));
				namedWindow("tracking", WINDOW_NORMAL);
				imshow("tracking", frame);

				//exit if ESC key is pressed
				if (waitKey(30) == 27)
					break;
				it++;
			} //main loop

			cout << it - 1 << "frames processed in " << 1000 * acum_t / t_freq
					<< " milliseconds." << endl;


	//release all resources

			cap.release();
			destroyAllWindows();
			waitKey(0);
		}
	}
	return 0;
}



