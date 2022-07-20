#include "explorationStep1.h"

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



void explorationStep1() {


	printf("Idan");

	string path = "C:/Users/admin/Desktop/fashion shoots/pictures claudia/IMG_5354.jpg";
	Mat img = imread(path, IMREAD_GRAYSCALE);
	int rows = img.rows;
	int cols = img.cols;


	cout << "Rws " << rows << endl;
	cout << "Cols " << cols << endl;

	Mat bgr[3];   //destination array
	split(img, bgr);//split source  

	Mat b = bgr[0];
	cout << b.rows << " " << b.cols << endl;

	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(img.rows);
	int n = getOptimalDFTSize(img.cols); // on the border add zero values
	copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
	dft(complexI, complexI);            // this way the result may fit in the source matrix

}
