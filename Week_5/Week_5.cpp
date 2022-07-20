

#include"Utilis_5.h"


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
#include<list>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;



int main()
{
    string path = "../../digital_images_week5_quizzes_noisy.jpg";

    string imgOrg = "../../digital_images_week5_quizzes_original.jpg";
    Mat imgNoise =read_file(path, "None", false, 2000, false);
    Mat imgOrig = read_file(imgOrg, "None", false, 2000, false);

    blur_Median(imgNoise, imgOrig,5, true);
    int pendingChange;

}
