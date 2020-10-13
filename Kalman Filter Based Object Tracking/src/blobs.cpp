
#include "blobs.hpp"
#include "kalmanFilter.hpp"
#include <cmath>
#include<vector>
using namespace std;
/**
 *	Draws blobs with different rectangles on the image 'frame'. All the input arguments must be
 *  initialized when using this function.
 *
 * \param frame Input image
 * \param pBlobList List to store the blobs found
 * \param labelled - true write label and color bb, false does not wirite label nor color bb
 *
 * \return Image containing the draw blobs. If no blobs have to be painted
 *  or arguments are wrong, the function returns a copy of the original "frame".
 *
 */
Mat paintBlobImage(cv::Mat frame, std::vector<cvBlob> bloblist, bool labelled) {
	cv::Mat blobImage;
	//check input conditions and return original if any is not satisfied
	//...

	frame.copyTo(blobImage);
	//required variables to paint
	//...
	cvBlob blob;

	//paint each blob of the list
	for (int i = 0; i < bloblist.size(); i++) {
		blob = bloblist[i]; //get ith blob	}

		Point p1 = Point(blob.x, blob.y);
		Point p2 = Point(blob.x + blob.w, blob.y + blob.h);

		if (labelled) {

			rectangle(blobImage, p1, p2, Scalar(255, 255, 255), 1, 8, 0);

		}




			//putText(blobImage, '', p1, FONT_HERSHEY_SIMPLEX, 1, color);

	} 
	//destroy all resources (if required)
	//...

	//return the image to show
	return blobImage;
}


/**
 *	Blob extraction from 1-channel image (binary). The extraction is performed based
 *	on the analysis of the connected components. All the input arguments must be
 *  initialized when using this function.
 *
 * \param fgmask Foreground/Background segmentation mask (1-channel binary image)
 * \param bloblist List with found blobs
 *
 * \return Operation code (negative if not succesfull operation)
 */
int extractBlobs(cv::Mat fgmask, std::vector<cvBlob> &bloblist,
		int connectivity) {
	//initialize rect
	Rect rect;
	//define new value that is gonna put by floodfil function
	Scalar newVal = Scalar(20);
	int blob_id = 0;
	Mat aux; // image to be updated each time a blob is detected (blob cleared)
	//create auxiliary image for connected component analysis
	fgmask.convertTo(aux, fgmask.type());

	//clear blob list (to fill with this function)
	bloblist.clear();
	for (int i = 0; i < aux.cols; i++) {
		for (int j = 0; j < aux.rows; j++) {
			if (aux.at<uchar>(Point(i, j)) == 255) {

				floodFill(aux, cv::Point(i, j), newVal, &rect, connectivity);
				cvBlob blob = initBlob(blob_id, rect.x, rect.y, rect.width,
						rect.height);
				//push into the list
				bloblist.push_back(blob);
				//increment blob id
				blob_id += 1;

			}
		}
	}

	//return OK code
	return 1;
}






int removeSmallBlobs(std::vector<cvBlob> bloblist_in,
		std::vector<cvBlob> &bloblist_out, int min_width, int min_height) {

	//clear blob list (to fill with this function)
	bloblist_out.clear();
	for (int i = 0; i < bloblist_in.size(); i++) {
		cvBlob blob_in = bloblist_in[i]; //get ith blob
		///&& blob_in.h < 100 && blob_in.w < 100
		if (blob_in.h >= min_height && blob_in.w >= min_width) {
			bloblist_out.push_back(blob_in);
				}

	}
			return 1;
		}
Point extractCenterOfLargestBlob(std::vector<cvBlob> &bloblist_out, int &x,
		int &y) {

	//double blob_area[bloblist_out.size()];

	int area = 0;

	for (int i = 0; i < bloblist_out.size(); i++) {
		
		if (area < int(bloblist_out[i].w * bloblist_out[i].h)) {
			cvBlob blob_in = bloblist_out[i];
			x = bloblist_out[i].x + ((bloblist_out[i].h) / 2);
			y = bloblist_out[i].y + ((bloblist_out[i].w) / 2);

			area = (blob_in.w * blob_in.h);

		}

	}


	return Point(x, y);
}




void applyMorphologicalOp(Mat fgmask) {
	

	//Define kernel size factor
	int kernel_size = 1;
	//Define the kernel
	Mat element = getStructuringElement(MORPH_RECT,
			Size(kernel_size + 2, kernel_size + 2),
			Point(kernel_size, kernel_size));
	//Apply opening operation
	morphologyEx(fgmask, fgmask, MORPH_OPEN, element);
	//Apply closing operation
	//morphologyEx(fgmask, fgmask, MORPH_CLOSE, element);

}

void drawTrejectory(cv::Mat frame, std::vector<Point> trajectoryPoints,
		Scalar color, int size, bool line) {

	///DRAW PREDICTED TREJECTORY
	vector<Point>::iterator ic;
	Point &temp = trajectoryPoints[0];
	for (ic = trajectoryPoints.begin(); ic != trajectoryPoints.end(); ic++) {
		Point p1;
		p1 = *ic;
		cv::circle(frame, p1, size, color, size);

		if (line) {
			if (p1.x - temp.x < 150) {
				cv::line(frame, temp, p1, color, 2);
			}
			temp = p1;
		}
	}

}




