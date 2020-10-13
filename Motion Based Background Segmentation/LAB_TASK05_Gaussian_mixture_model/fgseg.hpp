/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1.0: Background Subtraction - Unix version
 *	fgesg.hpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 */

#include <opencv2/opencv.hpp>

#ifndef FGSEG_H_INCLUDE
#define FGSEG_H_INCLUDE

using namespace cv;
using namespace std;

namespace fgseg {


	//Declaration of FGSeg class based on BackGround Subtraction (bgs)
	class bgs{
	public:

		//constructor with parameter "threshold"
	bgs(double threshold, double weight_threshold, bool rgb, double alpha);

		//destructor
		~bgs(void);

		//method to initialize bkg (first frame - hot start)
		void init_bkg(cv::Mat Frame);

		//method to perform BackGroundSubtraction
		void bkgSubtraction(cv::Mat Frame);

		//method to detect and remove shadows in the binary BGS mask
		void removeShadows();

		//returns the BG image
		cv::Mat getBG(){return _bkg;};

		//returns the DIFF image
		cv::Mat getDiff(){return _diff;};

		//returns the BGS mask
		cv::Mat getBGSmask(){return _bgsmask;};

		//returns the binary mask with detected shadows
		cv::Mat getShadowMask(){return _shadowmask;};

		//returns the binary FG mask
	cv::Mat getFGmask() {
		return _fgmask;
	}
	;


		//ADD ADITIONAL METHODS HERE
		//...

	void updateWeights(double weight[], int row, int col, double alpha);// to update the weights of non selected weights
	void updateBackgroundandMask(int col, int row, double weight_array[]); // to update the background

	private:
		cv::Mat _bkg; //Background model
		cv::Mat	_frame; //current frame
		cv::Mat _diff; //abs diff frame
		cv::Mat _bgsmask; //binary image for bgssub (FG)
		cv::Mat _shadowmask; //binary mask for detected shadows
		cv::Mat _fgmask; //binary image for foreground (FG)

	bool _rgb;
	double _alpha;
	double _threshold;
	float _weight_threshold; // this is the threshold for the sum of the weight for the Gaussian mixture model
		//ADD ADITIONAL VARIABLES HERE
		//...

	const static int num_gaussians = 2;	// to change the number of gaussian used change this value
	cv::Mat _mu[num_gaussians];
	cv::Mat sigma[num_gaussians];
	cv::Mat weight[num_gaussians];



	};//end of class bgs

}//end of namespace

#endif
