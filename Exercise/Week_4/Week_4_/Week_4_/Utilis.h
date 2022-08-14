#pragma once
// Utilis

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
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;


template<typename T>
static void show(T vec)
{
	cout << vec;
}

template<typename T>
void show_vec(std::vector<T> vec)
{
    int size = vec.size();
    if (size <= 0) {
        std::cout << "empty folder";
        return;
    }
    std::cout << '{';
    for (int l = 0; l < size - 1; l++) {
        show(vec[l]);
        std::cout << ',';
    }
    show(vec[size - 1]);
    std::cout << '}';
}

//template<typename T>
vector<string> get_files_folder(string type, string Path) {
    int size = type.size();

    fs::path path{ u8"愛.txt" };
    std::u8string path_string{ path.u8string() };

    vector<string> vecNames;
    for (const auto& file : fs::directory_iterator(Path)) {
        string b = fs::absolute(file.path()).string();

        string r = b.substr(b.length() - size);
        if (r == type) {
            vecNames.push_back(b);
        }
    }
    show_vec(vecNames);
    if (vecNames.size() > 0) {
        return vecNames;
    }
}

void path_toMat(string type, string Path, vector<Mat>& frames_Global) {
    vector<string> fileNamesLoc = get_files_folder(type, Path);
    for (int i = 0; i < fileNamesLoc.size(); i++) {
        frames_Global.push_back(imread(fileNamesLoc[i]));
    }
}

string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}
Mat drawBlock(Point UpperLeft,Point ButtomRight, Mat img1) {
    // Setup a rectangle to define your region of interest
   Mat cropped;
   Point new_x (UpperLeft.x, UpperLeft.y);
   Point new_y(ButtomRight.x - UpperLeft.x, ButtomRight.y - UpperLeft.y);
   Mat img_new = img1;
   img1(Rect(UpperLeft.x, UpperLeft.y, ButtomRight.x - UpperLeft.x, ButtomRight.y - UpperLeft.y)).copyTo(cropped);
   //rectangle(img_new, new_x, new_y, cv::Scalar(0, 255, 0));

   return cropped;

    // Copy the data into new matrix
}

double MSE(Mat b1, Mat b2, int m, int h) {
    Mat b1f = b1.reshape(1, 1);
    Mat b2f = b2.reshape(1, 1);
    
    Mat dst;
    absdiff(b2f, b1f, dst);
    return sum(dst)[0];
}
double MSE(Mat b1, Mat b2) {
    Mat b1f = b1.reshape(1,1);
    Mat b2f = b2.reshape(1, 1);

    Mat dst;
    Mat output = b1f.t() - b2f.t();
    cv::pow(output, 2, dst);
    double mse = sum(dst)[0];
    //cout << sum(dst)[0] << endl;
    return mse;
}

tuple<Mat, vector<Point>, double>  extensiveBlockMatchingSearch(Mat img1, Mat img2,int M,int N, vector<Point> points) {
    
    Mat most_similar;
    vector<Point> MostSimilarPOints;
    double best_MAE =numeric_limits<int>::max();

    Point u_l = points[0]; Point d_r= points[1];

    Point p(M, M);

    Mat given_region = drawBlock(u_l, u_l+p, img1);
    int row_search = img1.rows - 32;
    int col_search = img1.cols-32;

    for (int i = 0; i < row_search; i++) {
        for (int j = 0; j < row_search; j++) {

            Point cur_x(i, j);
            Point s(i+32, j+32);
            Mat cur_block = drawBlock(cur_x, s, img2);

                    // get MSe
        double curr_MSE = MSE(cur_block, given_region, M, N);
        if (curr_MSE < best_MAE) {
            best_MAE = curr_MSE;
            most_similar = cur_block;
            MostSimilarPOints.clear();
            MostSimilarPOints.push_back(s); MostSimilarPOints.push_back(cur_x);
        
    }
        }
    }

    cout << " Most similar: " << most_similar << endl;
    cout << " At X: " << MostSimilarPOints[0] << endl;
    cout << " Most similar: " << MostSimilarPOints[1] << endl;
    cout << " best_MAE: " << best_MAE << endl;


    return make_tuple(most_similar, MostSimilarPOints, best_MAE);
}


float PSNR(Mat img1, Mat img2, int r) {
    // PSNR = 10 log_10 (R^2 / MSE)

    double MSEval = MSE(img1, img2);

    auto PSNR = 10 * log10((r * r) / MSEval);
    return PSNR;
}