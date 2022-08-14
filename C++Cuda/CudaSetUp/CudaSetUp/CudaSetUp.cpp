// CudaSetUp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
//
#include<opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <iostream>
#include<opencv2/core/cuda.hpp>
#include<opencv2/dnn.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/dnn/all_layers.hpp>
#include <filesystem>
#include <iterator>
#include <fstream>

namespace fs = std::filesystem;
using namespace std;
using namespace cv::dnn;
using namespace cv;

#include "Utilis.h"




void main()
{
    cout << "OpenCV version : " << CV_VERSION << endl;

    String path = "D:/github_/Image_Video_Processing/C++Cuda/CudaSetUp/CudaSetUp/Models_/";
    String path2 = "COCO_labels.txt";
    string graph = "frozen_inference_graph.pb";
    string ModelName = "ssd_mobilenet_v2_coco_2018_03_29.pbtxt";
    string yolo = "C:/Users/admin/Downloads/yolov5n.onnx";

    string s = "C:/Users/admin/Downloads/ObjectDetector-OpenCV-main/frozen_inference_graph.pb";
    string full_ = "C:/Users/admin/Downloads/ObjectDetector-OpenCV-main/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";


    string YoloTry = "D:/github_/Image_Video_Processing/C++Cuda/CudaSetUp/CudaSetUp/Models_/Yolo5/yolov5l.pt";
    file_to_string(path, true);
    vector<string> fileNames= folderToStringName(path, true);

    vector<string> class_names;
    cout << fileNames[2] << endl;
    file_to_string_2(path, path2, false);

    //auto net = readNetFromTensorflow(s, full_);
    auto net = readNetFromTorch(YoloTry);
    
    if (net.empty())
    {
        cout << "check net file path\n";
        return;
    }
    else {
        cout << "Model Loaded" << endl;
    }
    
    VideoCapture cap(0);

    //net.setPreferableBackend(DNN_BACKEND_CUDA);
    //net.setPreferableTarget(DNN_BACKEND_CUDA);

    float minConfident_score = 0.6;

    while (cap.isOpened()) {

        Mat img; 
        bool isOpen = cap.read(img);

        if (!isOpen) {
            cout << "Not Open/ Check Loading" << endl;
            break;
        }

        int img_h = img.cols;
        int img_w = img.rows;


        auto start = getTickCount(); // for comparison GPU GPU

        Mat blob = blobFromImage(img, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5),
            true, false);
        //create blob from image
        net.setInput(blob);
        //forward pass through the model to carry out the detection
        Mat output = net.forward();


        //Mat blob = blobFromImage(img, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5), true, false); // image size, substract img value, 

        //
        //net.setInput(blob); // do the forward pass
        //Mat output = net.forward(); // with the model input we specified in the previous line

        auto end = getTickCount(); // end time 

        //cout << output.size << endl;
        Mat result(output.size[2], output.size[3], CV_32F, output.ptr<float>());

        for (int i = 0; i < result.rows; i++) {

            /// <summary>
            /// To note, the output of the result comes in the form of 
            /// [class id, confidence, int box_x, int box_y, int width_box, int height_box ]
            /// </summary>
            int class_id = int(result.at<float>(i, 1));
            int prob_confid = float(result.at<float>(i, 2));
            
            //cout << prob_confid << endl;
            // check if the confidence is better than specified min

            if (prob_confid > minConfident_score) {
                int bb_x = int(result.at<float>(i, 3)*img.cols);
                int bb_y = int(result.at<float>(i, 4)*img.rows);

                int intbb_width = int(result.at<float>(i, 5) * img.cols - bb_x); // check
                int intbb_height = int(result.at<float>(i, 6) * img.rows - bb_y); // check

                rectangle(img, Point(bb_x, bb_y), Point(bb_x+ intbb_width, bb_y+ intbb_height), Scalar(0, 0, 255), -1); // check
                cout << bb_x << endl;
                cout << bb_y << endl;
                string name_ = fileNames[class_id - 1] + " " + to_string(int(prob_confid));
                putText(img, name_,Point(bb_x,bb_y-10), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0,255,0),2);
            }

        }

        auto total_time = (end - start) / getTickFrequency();
        //cout << total_time << endl;
        //putText(img, total_time, )

        imshow("Display", img);

        int k = waitKey(10);
        if (k == 113) {
            break;
        }
    }
    cap.release();
    destroyAllWindows();

  
}

