// Week_4_.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <cstdlib>
#include <string>



#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/video/background_segm.hpp>

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
#include <iostream>


#include <sys\stat.h>
#include <iostream>
#include <direct.h>
#include <conio.h>

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include "Utilis.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;



int main() {
    string path = "D:/github_/Image_Video_Processing/Exercises";
    
    cv::namedWindow("Original", cv::WINDOW_NORMAL);
    cv::namedWindow("ROC", cv::WINDOW_NORMAL);

    int size = 32;
    Point bl(65, 81);
    Point br(96, 112);

    vector<Point> points;
    points.push_back(bl);  points.push_back(br);

    vector<Mat> frames;
    path_toMat("jpg", path, frames);
    

    extensiveBlockMatchingSearch(frames[0], frames[1], size, size, points);

}

    

