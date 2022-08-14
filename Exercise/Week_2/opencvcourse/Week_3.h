//#pragma once
//
//
//#include "Utilis.h"
//#include <string>
//#include <math.h>
//#include <opencv2/opencv.hpp>
//#include "opencv2/core.hpp"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//
//using namespace std;
//string GLOBALPATH = "../../PctureExercise/Week_3/digital_images_week3_quizzes_original_quiz.jpg";
//string FILE_NAME = "Week_3_quiz";
//
//class Week_3
//{
//private:
//	Point anchor;
//	int delta;
//	int ddepth;
//	Mat IMGBASE;
//public:
//	Week_3(int delta, int ddepth, Point an) {
//		delta = delta;
//		ddepth = ddepth;
//		anchor = an;
//	}
//	
//
//	void desplay_img();
//	void load_img_to_Memory();
//	Mat downsample(Mat img, int factor, int start);
//	void processing();
//	Mat  insert_zero_into_low_reso(Mat img, int height, int width);
//	Mat up_downSampling(Mat img, int factor);
//	string getPath()
//	{
//		return GLOBALPATH;
//	}
//	Mat read_image();
//
//};