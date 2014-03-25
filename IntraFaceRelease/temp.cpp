

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <time.h>
#include <intraface/FaceAlignmentDetect.h>
#include <intraface/FaceAlignmentTrack.h>
#include <intraface/XXDescriptor.h>

using namespace std;

bool compareRect(cv::Rect r1, cv::Rect r2) { return r1.height < r2.height; }

// 3 modes: REALTIME,IMAGE,VIDEO
#define IMAGE

bool drawing_box = true;
cv::Mat X;
cv::Rect box;

void draw_box(Mat* img, cv::Rect rect ){
	cv::rectangle( img, rect, cv::Scalar(0,255,0));
}

// Implement mouse callback
void my_mouse_callback( int event, int x, int y, int flags, void* param ){
	Mat* image = (Mat*) param;

	switch( event ){
		case CV_EVENT_MOUSEMOVE: 
			if( drawing_box ){
				box.width = x-box.x;
				box.height = y-box.y;
			}
			break;

		case CV_EVENT_LBUTTONDOWN:
			drawing_box = true;
			box = cv::Rect( x, y, 0, 0 );
			break;

		case CV_EVENT_LBUTTONUP:
			drawing_box = false;
			if( box.width < 0 ){
				box.x += box.width;
				box.width *= -1;
			}
			if( box.height < 0 ){
				box.y += box.height;
				box.height *= -1;
			}
			draw_box( image, box );
			float score;
			fad.Detect(*image,box,X,score);

			for (int i = 0 ; i < X.cols ; i++)
				cv::circle(*image,Point((int)X.at<float>(0,i), (int)X.at<float>(1,i)), 2, Scalar(0,255,0), -1);

			break;
	}
}


int main()
{
	string detectionModel("../models/DetectionModel-v1.5.yml");
#if defined(REALTIME) || defined(VIDEO)
	string trackingModel("../models/TrackingModel-v1.9.yml");
#endif
	string faceDetectionModel("../models/haarcascade_frontalface_alt2.xml");

	INTRAFACE::XXDescriptor xxd(8,4);
	INTRAFACE::FaceAlignmentDetect fad(detectionModel, &xxd, true);
	if (!fad.Initialized()) {
		cerr << "FaceAlignmentDetect cannot be initialized." << endl;
		return -1;
	}

#if defined(REALTIME) || defined(VIDEO)
	INTRAFACE::FaceAlignmentTrack fat(trackingModel, &xxd);
	if (!fat.Initialized()) {
		cerr << "FaceAlignmentTrack cannot be initialized." << endl;
		return -1;
	}
#endif	
	cv::CascadeClassifier face_cascade;
	if( !face_cascade.load( faceDetectionModel ) )
	{ 
		cerr << "Error loading face detection model." << endl;
		return -1; 
	}
	vector<cv::Rect> faces;

#ifdef REALTIME
	// use the first camera it finds
	cv::VideoCapture cap(0); 
#endif 

#ifdef VIDEO
	string filename("../data/vid.wmv");
	VideoCapture cap(filename); 
#endif

#if defined(REALTIME) || defined(VIDEO)
	if(!cap.isOpened())  
		return -1;
	//cap.set(CV_CAP_PROP_FRAME_WIDTH,1280);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT,720);
	

	int key = 0;
	bool isDetect = true;
	bool eof = false;
	float score, notFace = 0.3;
	cv::Mat X,X0;
	clock_t time_e;
	char frameRate[50];
	float frame_t;

	namedWindow("Test");

	while (key!=27) // Press Esc to quit
	{
		time_e = clock();
		Mat frame;
		cap >> frame; // get a new frame from camera
		if (isDetect)
		{
			face_cascade.detectMultiScale(frame, faces, 1.2, 2, 0, Size(50, 50));
			if (faces.empty()) {
				cv::imshow("Test",frame);
				key = waitKey(5);
				continue ;
			}
			// face alignment
			if (fad.Detect(frame,*max_element(faces.begin(),faces.end(),compareRect),X0,score) != INTRAFACE::IF_OK)
				break;
			isDetect = false;
		}
		else
		{
			// face alignment
			if (fat.Track(frame,X0,X,score) != INTRAFACE::IF_OK)
				break;
			X0 = X;
		}
		if (score < notFace) // detected face is not reliable
			isDetect = true;
		else
		{
			// plot facial landmarks
			for (int i = 0 ; i < X0.cols ; i++)
				cv::circle(frame,Point((int)X0.at<float>(0,i), (int)X0.at<float>(1,i)), 2, Scalar(0,255,0), -1);
		}
		time_e = clock() - time_e;
		frame_t = (float)CLOCKS_PER_SEC/time_e;
		memset(frameRate, 0, sizeof(char)*50);
		sprintf(frameRate, "%d FPS", fast_floor(frame_t));
		cv::putText(frame, frameRate, Point(50,50), CV_FONT_HERSHEY_PLAIN, 2, Scalar(255, 255, 255), 2);
		cv::imshow("Test",frame);	
		key = cv::waitKey(5);
	}
#endif

#ifdef IMAGE

	float score, notFace = 0.3;
	cv::Mat X;
	
	string filename("../data/pic.jpg");
	cv::Mat frame  = cv::imread(filename);

	// Set up the callback
	string winname("Test");
	cvSetMouseCallback( winname.c_str(), my_mouse_callback, (void*) &frame);
	cv::namedWindow(winname);
	// Main loop
	while( true ) {

		if( drawing_box ) 
			draw_box( temp, box );
		cv::imshow(winname,frame);

		if( cvWaitKey( 15 )==27 ) 
			break;
	}

	

#endif


	return 0;

}






