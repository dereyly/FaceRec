///////////////////////////////////////////////////////////////////////////////////////////////////////
/// DemoDetector.cpp
/// 
/// Description:
/// This program shows you how to use FaceAlignment class in detecting facial landmarks on one image. 
///
/// There are two modes: INTERACTIVE, AUTO.
///
/// In the INTERACTIVE mode, the user is asked to create a draggable rectangle to locate one's face. 
/// To obtain good performance, the upper and lower boundaries need to exceed one's eyebrow and lip. 
/// For examples of good input rectangles, please refer to "../data/good_input_rect.jpg".
///
/// In the AUTO mode, the faces are found through OpenCV face detector.
///
/// Note that the initialization is optimized for OpenCV face detector. However, the algorithm is not
/// very sensitive to initialization. It is possible to replace OpenCV's with your own face detector. 
/// If the output of your face detector largely differs from the OpenCV's, you can add a constant offset
/// to the output of your detector using an optional parameter in the constructor of FaceAlignment.
/// See more details in "FaceAlignment.h".
///
/// Dependencies: None. OpenCV DLLs and include folders are copied.
///
/// Author: Xuehan Xiong, xiong828@gmail.com
///
/// Creation Date: 10/15/2013
///
/// Version: 1.0
///
/// Citation: 
/// Xuehan Xiong, Fernando de la Torre, Supervised Descent Method and Its Application to Face Alignment. CVPR, 2013
///////////////////////////////////////////////////////////////////////////////////////////////////////


#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
//#include "opencv2/contrib/contrib.hpp"
#include "../LBPH/facerec.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <intraface/FaceAlignment.h>
#include <intraface/XXDescriptor.h>
#include <math.h>
#include <direct.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "DetectParts.h"

using namespace std;
using namespace cv;
//using cv::Rect;
//using cv::Mat;
//using cv::CascadeClassifier;
//using cv::Scalar;
//using cv::Size;
//using cv::FaceRecognizer;
string cascadeName = "c:\\libs\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt.xml";

// 2 modes: AUTO, INTERACTIVE
#define AUTO




int main(int argc, const char *argv[]) 
{
	bool is_demo=false;
	//string fn_csv1="d:\\ImageDB\\FACE\\CroppedYale\\faces_train.ext";
	//string fn_csv2="d:\\ImageDB\\FACE\\CroppedYale\\faces_test.ext";
	//string fn_csv1="d:\\ImageDB\\FACE\\YaleFull\\faces_train2.ext";
	//string fn_csv2="d:\\ImageDB\\FACE\\YaleFull\\faces_test2.ext";
	string fn_csv1="d:\\ImageDB\\FACE\\LFW\\faces_train_sm.ext";
    string fn_csv2="d:\\ImageDB\\FACE\\LFW\\faces_test_sm.ext";

	PartDetector pdet;
	//pdet.Init("c:\\libs\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt.xml","../models/DetectionModel-v1.5.bin","../models/TrackingModel-v1.10.bin");
	pdet.Init("../models/haarcascade_frontalface_alt2.xml","../models/DetectionModel-v1.5.bin","../models/TrackingModel-v1.10.bin");
	vector<Mat> img_train,img_test;
	vector<int> lbl_train,lbl_test;
	vector<vector<Rect>> fpoints_tr,fpoints_tst;

	const int64 start0 = cvGetTickCount();
	//pdet.ExtractCSVFaces(fn_csv1, img_train, lbl_train, fpoints_tr);
	pdet.ExtractCSVFaces(fn_csv2, img_test, lbl_test, fpoints_tst);


	const int64 finish0 = cvGetTickCount();
	cout << "alighn time =" << (float)(finish0 - start0) / (float)(cvGetTickFrequency() * 1000000.0)<< endl;
  
	Ptr<libfacerec::FaceRecognizer> model;

	model = libfacerec::createLBPHFaceRecognizer(2,6,4,4,123.0);
	//model = libfacerec::createLBPHFaceRecognizer(4,8,7,7,123.0);
	//model =libfacerec::createFisherFaceRecognizer(); //Ошибка с етм что изображение разрывно, возникает с введением новых параметров
	//model = libfacerec::createEigenFaceRecognizer();
	const int64 start = cvGetTickCount();
    //model->train_parts(img_train, lbl_train,fpoints_tr);
	//model->train(img_train, lbl_train);
	model->load("data_part.bin");
	const int64 finish = cvGetTickCount();
	cout << "train time =" << (float)(finish - start) / (float)(cvGetTickFrequency() * 1000000.0)<< endl;
    // The following line predicts the label of a given
    // test image:
	double acc=0;
	vector<int> lbls;
	//char tmp[100];
	if (is_demo)
	{
		VideoCapture cap(0); // open the default camera
		if(!cap.isOpened())  // check if we succeeded
			return -1;

		//Mat edges;
		namedWindow("faces",1);
		char tmp[100];
		
		for(;;)
		{
			Mat frame;
			cap >> frame; // get a new frame from camera
			vector<vector<Rect>> fpoints;
			vector<vector<Rect>> face_parts;
			vector<Mat> faces_crop;
			pdet.ExtractAllFaces(frame,faces_crop,fpoints,face_parts);
			for (int i=0;i<face_parts.size();i++)
			{
				int lbl_out = model->predict(faces_crop[i]);
				//imshow("faces_crop", faces_crop[i]);
				sprintf(tmp,"face_id=%d",lbl_out);
				for (int j=0;j<face_parts[i].size();j++)
					//circle(frame,Point(face_parts[i][j].x, face_parts[i][j].y), 1, Scalar(0,255,0), -1);
					//rectangle(frame,face_parts[i][j],Scalar(0,255,0));
					rectangle( faces_crop[i],fpoints[i][j],Scalar(0,255,0));
				for (int j=0;j<face_parts[i].size();j++)
					circle(frame,Point(face_parts[i][j].x, face_parts[i][j].y), 1, Scalar(0,255,0), -1);

				if (lbl_out>=62)
				{
					rectangle(frame,face_parts[i][0],Scalar(0,255,0));
					if (lbl_out==62)
						putText(frame,"Nikolay", Point(face_parts[i][0].x,face_parts[i][0].y), FONT_HERSHEY_SIMPLEX,0.7, Scalar(255,0,0));
					if (lbl_out==63)
						putText(frame,"Tatiana", Point(face_parts[i][0].x,face_parts[i][0].y), FONT_HERSHEY_SIMPLEX,0.7, Scalar(255,0,0));
					if (lbl_out==64)
						putText(frame,"Vladimir", Point(face_parts[i][0].x,face_parts[i][0].y), FONT_HERSHEY_SIMPLEX,0.7, Scalar(255,0,0));

				}
				else
				{
					rectangle(frame,face_parts[i][0],Scalar(255,0,0));
					putText(frame,"unknown", Point(face_parts[i][0].x,face_parts[i][0].y), FONT_HERSHEY_SIMPLEX,0.7, Scalar(255,0,0));
				}
				imshow("faces crop", faces_crop[i]);
				//cout << " " << lbl_out; //0.275
			}
			//cout << " " << face_parts.size();
			imshow("faces", frame);
			if(waitKey(30) >= 0) break;
		}
		return 0;
	}
	const int64 start2 = cvGetTickCount();
	double sum_true=0, sum_false=0;
	double err1=0,err2=0;
	for (int i=0;i<img_test.size();i++)
	{
		int lbl_out;// = model->predict_parts(img_test[i],fpoints_tst[i]);
		double distt;
		model->predict_parts(img_test[i],lbl_out,distt,fpoints_tst[i]);
		//int lbl_out = model->predict(img_test[i]);
		lbls.push_back(lbl_out);
 		if (lbl_out==lbl_test[i])
 		{
 			acc++;
 			sum_true+=distt;
 		}
 		else
 			sum_false+=distt;

// 		else 
// 			cout << i << "->" << distt << endl;
		if (lbl_out!=lbl_test[i] && distt<11)
			err1++;
		if (distt>11)
			err2++;
	}
	cout << "true dist=" << sum_true/img_test.size() << " false dist="  << sum_false/((double)img_test.size()-acc) << endl;
	//model->save("data4.bin");
	const int64 finish2 = cvGetTickCount();
	cout << "test time =" << (float)(finish2 - start2) / (float)(cvGetTickFrequency() * 1000000.0)<< endl;

	acc/=img_test.size();
	cout << "-------------> accuracy=" << acc <<  " n_test_files=" << img_test.size() <<endl;
	err1/=img_test.size();
	err2/=img_test.size();
	cout << "-------------> err1=" << err1 <<  " err2=" <<err2 <<endl;
   // model->set("threshold", 0.0);
	//model->save("data_part.bin");
	
    return 0;
}