/* Applied Video Analysis of Sequences (AVSA)
 *
 *	LAB2: Blob detection & classification
 *	Lab2.0: Sample Opencv project
 *
 *
 * Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es), Juan C. San Miguel (juancarlos.sanmiguel@uam.es)
 */

#include "blobs.hpp"
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

	//paint each blob of the list
	for (int i = 0; i < bloblist.size(); i++) {
		cvBlob blob = bloblist[i]; //get ith blob
		//...
		Scalar color;
		std::string label = "";
		switch (blob.label) {
		case 1:
			color = Scalar(255, 0, 0);
			label = "PERSON";
			break;
		case 3:
			color = Scalar(0, 255, 0);
			label = "CAR";
			break;
		case 4:
			color = Scalar(0, 0, 255);
			label = "OBJECT";
			break;
		default:
			color = Scalar(255, 255, 255);
			label = "UNKNOWN";
		}

		Point p1 = Point(blob.x, blob.y);
		Point p2 = Point(blob.x + blob.w, blob.y + blob.h);

		rectangle(blobImage, p1, p2, color, 1, 8, 0);
		if (labelled) {
			rectangle(blobImage, p1, p2, color, 1, 8, 0);
			putText(blobImage, label, p1, FONT_HERSHEY_SIMPLEX, 0.5, color);
		} else
			rectangle(blobImage, p1, p2, Scalar(255, 255, 255), 1, 8, 0);
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
	//check input conditions and return -1 if any is not satisfied

	//required variables
	//...

	//clear blob list (to fill with this function)
	bloblist_out.clear();
	for (int i = 0; i < bloblist_in.size(); i++) {
		cvBlob blob_in = bloblist_in[i]; //get ith blob
		if (blob_in.h >= min_height && blob_in.w >= min_width) {
			bloblist_out.push_back(blob_in);
		}
		// ...............................
		// void implementation (does not remove)

	}
	//destroy all resources
	//...
	//bloblist_out = bloblist_in;

	//return OK code
	return 1;
}



 /**
 *	Blob classification between the available classes in 'Blob.hpp' (see CLASS typedef). All the input arguments must be
 *  initialized when using this function.
 *
 * \param frame Input image
 * \param fgmask Foreground/Background segmentation mask (1-channel binary image)
 * \param bloblist List with found blobs
 *
 * \return Operation code (negative if not succesfull operation)
 */

// ASPECT RATIO MODELS
#define MEAN_PERSON 0.3950
#define STD_PERSON 0.1887

#define MEAN_CAR 1.4736
#define STD_CAR 0.2329

#define MEAN_OBJECT 1.2111
#define STD_OBJECT 0.4470

// end ASPECT RATIO MODELS


/////////////////////////////
// ASPECT RATIO MODELS REGENERATED FROM DATA SET WITH MATLAB SCRIPT (OPTIONAL TASK 02)
// Some of the values are further tuned. as the values we got from Dataset are not working properly.

#define MEAN_PERSON01_GENERATED 0.5329
#define STD_PERSON01_GENERATED  0.3032

#define MEAN_PERSON02_GENERATED 0.4050
#define STD_PERSON02_GENERATED  0.0303


#define MEAN_CAR_GENERATED 1.5
#define STD_CAR_GENERATED  0.2329

#define MEAN_OBJECT_GENERATED 1.0888
#define STD_OBJECT_GENERATED  0.2028
////////////////////////////


// distances
float ED(float val1, float val2) {
	return sqrt(pow(val1 - val2, 2));
}

float WED(float val1, float val2, float std)
{
	return sqrt(pow(val1 - val2, 2) / pow(std, 2));
}
//end distances
int classifyBlobs(std::vector<cvBlob> &bloblist) {
	//check input conditions and return -1 if any is not satisfied
	//...



 	//required variables for classification
	//...
	double ratios[3] = { };

	//classify each blob of the list
	for (int i = 0; i < bloblist.size(); i++) {
		cvBlob blob = bloblist[i]; //get i-th blob

		//Calculate the aspect ratio for current blob
		double aspect_ratio = double(blob.w) / double(blob.h);

		//Check the euclidean distance between current blob aspect ratio and Model's mean aspect ratio.
		ratios[0] = ED(aspect_ratio, MEAN_PERSON02_GENERATED);
		ratios[1] = ED(aspect_ratio, MEAN_CAR_GENERATED);
		ratios[2] = ED(aspect_ratio, MEAN_OBJECT_GENERATED);

		//Select the minimum distance from the list
		double *min = std::min_element(ratios, ratios + 3);

		//find the index of the minimum value in the list
		int x = std::distance(ratios, std::find(ratios, ratios + 3, *min));
		if (aspect_ratio >= 3.5) {
			x = 3;
		}
		//select the corresponding label for blob
		if (x == 0) {
			bloblist[i].label = PERSON;
		} else if (x == 1) {
			bloblist[i].label = CAR;
		} else if (x == 2) {
			bloblist[i].label = OBJECT;
		} else {
			bloblist[i].label = UNKNOWN;
		}
		//...

		// void implementation (does not change label -at creation UNKNOWN-)
	}

	//destroy all resources
	//...

	//return OK code
	return 1;
}

//stationary blob extraction function
/**
 *	Stationary FG detection
 *
 * \param fgmask Foreground/Background segmentation mask (1-channel binary image)
 * \param fgmask_history Foreground history counter image (1-channel integer image)
 * \param sfgmask Foreground/Background segmentation mask (1-channel binary image)
 *
 * \return Operation code (negative if not succesfull operation)
 *
 *
 * Based on: Stationary foreground detection for video-surveillance based on foreground and motion history images, D.Ortego, J.C.SanMiguel, AVSS2013
 *
 */

#define FPS 20 //check in video - not really critical
#define SECS_STATIONARY 5 // to set
#define I_COST 1 // to set // increment cost for stationarity detection
#define D_COST 7 // to set // decrement cost for stationarity detection
#define STAT_TH 0.5 // to set

int extractStationaryFG(Mat fgmask, Mat &fgmask_history, Mat &sfgmask) {

	//cout << fgmask << endl;
	int numframes4static = (int) (FPS * SECS_STATIONARY);

	//Make both matrix same type
	fgmask.convertTo(fgmask, fgmask_history.type());

	//Calculate the logical mask
	fgmask = fgmask / 255;

	//Create foreground history FHI.
	fgmask_history = fgmask_history + (I_COST * fgmask);
	fgmask_history = fgmask_history - ( D_COST * (1 - fgmask));
	
	//eliminate negative values from the matrix
	threshold(fgmask_history, fgmask_history, 0, 255, THRESH_TOZERO);

	//Normalize foreground history image
	Mat noramlized_fgmask_history = min(1, fgmask_history / numframes4static);


	//Threshold normalized FHI.
	sfgmask = (noramlized_fgmask_history >= STAT_TH);


	sfgmask = sfgmask * 255;


	return 1;

}



