#include "Week_3.h"
#include "Utilis.h"

#include <cstdlib>
#include <string>

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include<list>
#include <math.h>
#include <stdio.h>

using namespace cv;
using namespace std;



using namespace std;
string GLOBALPATH = "../../PctureExercise/Week_3/digital_images_week3_quizzes_original_quiz.jpg";
string FILE_NAME = "Week_3_quiz";

class Week_3
{
private:
	Point anchor;
	int delta;
	int ddepth;
	Mat IMGBASE;
public:
	Week_3(int delta, int ddepth, Point an) {
		delta = delta;
		ddepth = ddepth;
		anchor = an;
	}


	void desplay_img();
	void load_img_to_Memory();
	Mat downsample(Mat img, int factor, int start);
	void processing();
	Mat  insert_zero_into_low_reso(Mat img, int height, int width);
	Mat up_downSampling(Mat img, int factor);
	string getPath()
	{
		return GLOBALPATH;
	}
	Mat read_image();

};

Mat Week_3::read_image()
{
	return imread(GLOBALPATH);
}
void Week_3::desplay_img(){
	Mat img = read_image();
	imshow(FILE_NAME, img);
	waitKey(0);
}
Mat _kernel(int h, int w, string how) { // This I want to take from the other class
	if (how == "box") {
		return Mat::ones(h, w, CV_64F) / (h * w);
	}
}
Mat Week_3:: downsample(Mat img, int factor, int start) {
	int d_or = img.rows; int n_or  =img.cols;
	int d_new = factor * d_or; int n_new = n_or * factor;

	Mat output = Mat::zeros(d_new, n_new, CV_64F);

	for (int i = 0; i <= d_or; i++) {

		for (int j = 0; j < n_or; j++) {
			if (i >= start && j >= start) {
				output.at<double>(i * factor, j * factor) = img.at<double>(i, j);
			}
		}
	}
	return output;
}

Mat Week_3::insert_zero_into_low_reso(Mat img, int height, int width ) {
	Mat inset_zeros_mat = Mat::zeros(height, width, CV_64F);
	for (int i = 0; i <= height; i++) {

		for (int j = 0; j <= width; j++) {
			inset_zeros_mat.at<double>(i, j) = img.at<double>((i + 1) / 2, (j + 1) / 2);
		}
	}
	return inset_zeros_mat;
}
void Week_3::load_img_to_Memory(){
	IMGBASE = read_image();
}

Mat Week_3::up_downSampling(Mat img, int factor) {
	Mat dst;
	cv::resize(img, dst, Size(), factor, factor, INTER_LINEAR);
	return dst;
}

void Week_3:: processing() {
	int k = 3;
	anchor = Point(-1, -1);
	delta = 0;
	ddepth = -1;
	int ind = 0;
	Mat img,dst;
	Mat kernel = _kernel(k, k, "box");

	filter2D(IMGBASE, dst, ddepth, kernel, anchor, delta, BORDER_DEFAULT);

	// obtain down sanmpling

	Mat all_zero = Mat::zeros(379, 497, CV_64F);


	Mat_<double>kernel2d(3, 3);
	Mat convolved;
	Mat lowe_res = downsample(IMGBASE, 0.5, 2);
	Mat insert_zero = insert_zero_into_low_reso(lowe_res, 359, 379);
	kernel2d < 0.25, 0.5, 0.25, 0.5, 1, 0.5, 0.25, 0.5, 0.25;
	cv::namedWindow("Before", cv::WINDOW_NORMAL);
	cv::namedWindow("filter_2d", cv::WINDOW_NORMAL);

	filter2D(insert_zero, convolved, ddepth, kernel2d, anchor, delta, BORDER_DEFAULT);


}
