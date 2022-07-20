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

#include "Test.h"
#include "Question5_6_week_2.cpp"
#include "SpatialFiltering.cpp"
#include "Utilis.h"
#include "Week_3.cpp"

using namespace cv;
using namespace std;


void main() {

	//nn->defineKernek(4,4, "gaussian");

	// Question 5
	auto question_5 = new Question_5();
	string path_5 = "../../PctureExercise/Week_3/digital_images_week3_quizzes_original_quiz.jpg";

	Mat src = imread(path_5);

	//Mat imgreturn = question_5->importimage(path_5, 3);
	cv::namedWindow("Before", cv::WINDOW_NORMAL);
	cv::namedWindow("Kernel", cv::WINDOW_NORMAL);
	cv::namedWindow("filter_2d", cv::WINDOW_NORMAL);

	Mat_<float>kernel2d(3, 3);
	Mat_<float>fil2d(2, 2);
	kernel2d << 0.25, 0.5, 0.25; 0.5, 1, 0.5; 0.25, 0.5, 0.25;//< 0.25, 0.5, 0.25, 0.5, 1, 0.5, 0.25, 0.5, 0.25;
	fil2d << 1, 2, 2, 1;
	Mat conv2;
	filter2D(src, conv2, -1, kernel2d, Point(-1, 1));
		
	Mat costum2, kernek8u, convolved;
	
	kernel2d.convertTo(costum2, CV_8UC1);
	fil2d.convertTo(kernek8u, CV_8UC1);
	conv2.convertTo(convolved, CV_8UC1);



	imshow("Before", costum2);
	imshow("Kernel", kernek8u);
	imshow("filter_2d", convolved);

	//Week_3 funcweek_3 = Week_3(1, 0, Point(-1,1));
	waitKey(0);
}