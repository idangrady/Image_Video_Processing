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



class Question_5 {
private:
	Point anchor;
	double delta;
	int ddepth;
	int kernel_size;
	string window_name = "Window:";

public:

	Mat Kernel(int height, int width) {
		Mat Kernel =  Mat::ones(height, width, CV_64F) / (float)(height * width);
		cout << Kernel << endl;
		return Kernel;
	}
	double MSE(Mat &img1 ,Mat &img2) {

		float output;
		output = (1 / (img1.rows* img1.cols));

		int h_1 = img1.rows;
		int w_1 = img1.cols;
		Mat img1_1Dim = img1.reshape(1);
		Mat img2_1Dim = img2.reshape(1);

		Mat result_MSE = (img1_1Dim - img2_1Dim);
		Mat output_power;
		pow(result_MSE, 2, output_power); // ^2
		double MSEOutout = sum(output_power)[0] / (h_1 * w_1);;

		return MSEOutout;
	}

	float PSNR(Mat img1, Mat img2, int r) {
		// PSNR = 10 log_10 (R^2 / MSE)

		double MSEval = MSE(img1, img2);

		auto PSNR = 10 * log10((r * r) / MSEval);
		return PSNR;
	}

	Mat importimage(string path, int k) {

		Mat src = imread(path);

		if (src.empty())
		{
			printf(" Error opening image\n");
			printf(" Program Arguments: [image_name -- default lena.jpg] \n");
			return src;
		}

		Mat dst;

		anchor = Point(-1, -1);
		delta = 0;
		ddepth = -1;

		int ind = 0;

		Mat kernek2D = Kernel(k, k);
		kernel_size = 3;
		Mat kernel_ = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);

		filter2D(src, dst, ddepth, kernel_, anchor, delta, BORDER_DEFAULT);
		imshow(window_name, dst);
		double PSNR_val = PSNR(src, dst, 255);
		cout << "PSNR_val " << PSNR_val << endl;

		waitKey(0);
		return dst;

	}

};