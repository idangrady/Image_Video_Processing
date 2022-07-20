#pragma once
#include "../../../../../github_/Image_Video_Processing/Exercises/Week_4/Week_4_/Week_4_/Utilis.h"


using namespace cv;
using namespace std;
namespace fs = std::filesystem;



Mat read_file(string Path, string type, bool test, int waitTime, bool norm) {
	Mat output = imread(Path, CV_8SC1);
	Mat outputNormlize;
	Mat dst;
	cvtColor(output, dst, COLOR_RGB2GRAY);
	auto r = type2str(dst.type());


	cout << "Nat type: " << r << endl;

	if (norm) {
		normalize(output, outputNormlize, NORM_MINMAX);
		output = outputNormlize;
		//output = output / 255;
	}

	if (test) {
		imshow("Test", output);
		waitKey(waitTime);
	}
	return output;
}



void blur_Median(Mat imgNoise,Mat imgOrig, int greedySearch, bool print) {

	Mat img_i = imgNoise.clone();
	double PSNrOriginal = PSNR(imgNoise, imgOrig);
	cout << "PSNrOriginal " << PSNrOriginal << endl;

	for (int i = 3; i <= greedySearch; i = i + 2) {

		Mat dst; 
		medianBlur(img_i, dst, i);

		double psnrCur = PSNR(dst, imgOrig);
		if (print) {
			cout << "Median Blur " << i << " Size" << endl;
			cout << "psnrCur " << psnrCur << endl;
			
			imshow("dst", dst); waitKey(1500);

		}
		img_i = dst;
	}
}

