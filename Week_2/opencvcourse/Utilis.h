#pragma once
#include "Week_3.h"


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


static Mat make_kernel(int h, int w, string how) {
	if (how == "box") {
		return Mat::ones(h, w, CV_64F) / (h * w);
	}
}

