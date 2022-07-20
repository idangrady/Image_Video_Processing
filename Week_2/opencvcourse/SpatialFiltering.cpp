#include "SpatialFiltering.h"

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

using namespace cv;
using namespace std;

class SpatialFiltering_ {
public:
	Mat padding(Mat img, int k, int k_width, int k_height, string type) {

		Mat scr;
		img.convertTo(scr, CV_64FC1);		// convert to 1 channel
		int pad_rows, pad_colms;

		pad_rows = (k_width - 1) / 2;
		pad_colms = (k_height - 1) / 2;

		Mat img_outPadded;
		copyMakeBorder(scr, img_outPadded, 0, pad_rows, 0, pad_colms, BORDER_CONSTANT, Scalar::all(0));

		Mat pad_image(Size(scr.cols + 2 * pad_colms, scr.rows + 2 * pad_rows), CV_64FC1, Scalar(0));
		scr.copyTo(pad_image(Rect(pad_colms, pad_rows, scr.cols, scr.rows)));

		return scr;
	}

	Mat defineKernek(int K_Height, int K_Width, string type) {

		if (type == "box") {
			int n = K_Height * K_Width;

			Mat Kernel(K_Height, K_Width, CV_64FC1, Scalar(1 / n));				// 1/n for all places
		}

		else if (type == "gaussian")
		{
			int pad_rows = (K_Height - 1) / 2;
			int pad_cols = (K_Width - 1) / 2;
			Mat Kernel(K_Height, K_Width, CV_64FC1);				// We will fill the spaces for the guassian filters


			for (int i = -pad_cols; i <= pad_cols; i++) {

				for (int j = -pad_rows; i <= pad_rows; i++) {

					Kernel.at<double>(j + pad_rows, i + pad_cols) = exp(-(i * i + j * j) / 2);
				}

			}
			cout << Kernel << endl;
			for (int i = 0; i <= K_Height; i++) {
				for (int j = 0; j < K_Width; j++) {
					cout << Kernel.at<double>(i, j) << " ";;
				}
				cout << "" << endl;;
			}
			return Kernel;

		}
	}
};