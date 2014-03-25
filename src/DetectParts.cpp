

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
#include <fstream>
//#include <sstream>
#include "DetectParts.h"

int PartDetector::GetFirstFace( Mat& frame, vector<Rect> &out_rect)
{
	vector<cv::Rect> faces;
	bool write_debug=false;
	float score, notFace = 0.5,max_score=0;
	int ind_max_score=-1;
	// face detection
	m_cascade.detectMultiScale(frame, faces, 1.2, 2, 0, cv::Size(50, 50));
	/*vector<vector<Point>> fpoints_all;
	vector<Point> fpoints;*/
	vector<vector<Rect>> fpoints_all;
	vector<Rect> fpoints;

	for (int i = 0 ;i < faces.size(); i++) {
		//cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
		//rectangle(frame,faces[i],Scalar(255,0,0));
		// face alignment
		if (m_fa->Detect(frame,faces[i],m_FacePoints,score) == INTRAFACE::IF_OK)
		{

			if (score >= notFace) 
			{
				fpoints.push_back(faces[i]);
				for (int i = 0 ; i < m_FacePoints.cols ; i++)
					//fpoints.push_back(Point((int)X.at<float>(0,i), (int)X.at<float>(1,i)));
					fpoints.push_back(Rect((int)m_FacePoints.at<float>(0,i), (int)m_FacePoints.at<float>(1,i),0,0));
					//cv::circle(frame,cv::Point((int)X.at<float>(0,i), (int)X.at<float>(1,i)), 1, cv::Scalar(0,255,0), -1);
				if (score>max_score)
				{
					max_score=score;
					ind_max_score=i;
				}
			}
			
		}
		fpoints_all.push_back(fpoints);
	}
	if (ind_max_score==-1)
	{
		if (write_debug)
		{
			char tmp[100];
			sprintf(tmp,"%d.jpg",rand()%1000);
			imwrite(tmp,frame);
		}
		return -1;
	}
	out_rect=fpoints_all[ind_max_score];

	return 0;

}
int PartDetector::GetAllFaces( Mat& frame, vector<vector<Rect>> &out_rect)
{
	vector<cv::Rect> faces;
	bool write_debug=false;
	float score, notFace = 0.5,max_score=0;
	int ind_max_score=-1;
	// face detection
	m_cascade.detectMultiScale(frame, faces, 1.2, 2, 0, cv::Size(50, 50));
	/*vector<vector<Point>> fpoints_all;
	vector<Point> fpoints;*/
	vector<vector<Rect>> fpoints_all;
	
	

	for (int i = 0 ;i < faces.size(); i++) {
		vector<Rect> fpoints;
		//cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
		//rectangle(frame,faces[i],Scalar(255,0,0));
		// face alignment
		if (m_fa->Detect(frame,faces[i],m_FacePoints,score) == INTRAFACE::IF_OK)
		{

			if (score >= notFace) 
			{
				fpoints.push_back(faces[i]);
				for (int i = 0 ; i < m_FacePoints.cols ; i++)
					//fpoints.push_back(Point((int)X.at<float>(0,i), (int)X.at<float>(1,i)));
					fpoints.push_back(Rect((int)m_FacePoints.at<float>(0,i), (int)m_FacePoints.at<float>(1,i),0,0));
					//cv::circle(frame,cv::Point((int)X.at<float>(0,i), (int)X.at<float>(1,i)), 1, cv::Scalar(0,255,0), -1);
				if (score>max_score)
				{
					max_score=score;
					ind_max_score=i;
				}
				fpoints_all.push_back(fpoints);
			}
			
		}
		
	}
	if (ind_max_score==-1)
	{
		if (write_debug)
		{
			char tmp[100];
			sprintf(tmp,"%d.jpg",rand()%1000);
			imwrite(tmp,frame);
		}
		return -1;
	}
	out_rect=fpoints_all;

	return 0;

}
 void PartDetector::read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator ) 
 {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

 void PartDetector::read_csv_names(const string& filename, vector<string>& im_names, vector<int>& labels, char separator ) 
{
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if(!path.empty() && !classlabel.empty()) {
			im_names.push_back(path);
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}



Rect PartDetector::IncreaseRectSize(Rect src,double coeff, Size im_sz)
{
	Rect big_rect;
	int mx,my,w,h;
	w=src.width;
	h=src.height;
	mx=src.x+w/2;
	my=src.y+h/2;
	w*=coeff;
	h*=coeff;
	big_rect.x=std::max(0,mx-w/2);
	big_rect.y=std::max(0,my-h/2);
	int x2,y2;
	x2=std::min(im_sz.height,mx+w/2);
	y2=std::min(im_sz.width,my+h/2);
	big_rect.width=x2-big_rect.x;
	big_rect.height=y2-big_rect.y;
	return big_rect;
}

void PartDetector::StandartRectSize(vector<Rect> &rects,double scale_face_part, Size im_sz, bool is_lbph)
{
	//Rect big_rect;
	//double coeff=128.0/rects[0].width; //ToDO убрать константу потом
	for (int i=1;i<rects.size();i++)
	{
		
		int mx,my,w,h;
		w=rects[i].width;
		h=rects[i].height;
		mx=rects[i].x+w/2;
		my=rects[i].y+h/2;
		w=scale_face_part*rects[0].width;//+0.5;
		h=scale_face_part*rects[0].height;//+0.5;
		rects[i].x=std::max(0,mx-w/2);
		rects[i].y=std::max(0,my-h/2);
		if (is_lbph)
		{
			int x2,y2;
			y2=std::min(im_sz.height,mx+h/2);
			x2=std::min(im_sz.width,my+w/2);
			rects[i].width=x2-rects[i].x;
			rects[i].height=y2-rects[i].y;
		}
		else
		{
			rects[i].width=w;
			rects[i].height=h;
			int dx,dy;
			dx=im_sz.width<=rects[i].x+rects[i].width ? (rects[i].x+rects[i].width)-im_sz.width+1:0;
			dy=im_sz.height<=rects[i].y+rects[i].height ? (rects[i].y+rects[i].height)-im_sz.height+1:0;
			rects[i].x-=dx;
			rects[i].y-=dy;
		}
	}
}

Rect PartDetector::GetMidRect(vector<Rect> &fpoints,vector<int> IX)
{
	Rect out;
	for (int i=0; i<IX.size(); i++)  // индексы съехали на 1 из за того что первый элемент все лицо целиком
		IX[i]++;
	int min_x,max_x;//,max_y,min_y;
	min_x=fpoints[IX[0]].x;
	max_x=fpoints[IX[0]].x;
	//min_y=fpoints[IX[0]].y;
	//max_y=fpoints[IX[0]].y;
	for (int i=0; i<IX.size(); i++)
	{
		out.x+=fpoints[IX[i]].x;
		out.y+=fpoints[IX[i]].y;

		if (max_x<fpoints[IX[i]].x)
			max_x=fpoints[IX[i]].x;

		if (min_x>fpoints[IX[i]].x)
			min_x=fpoints[IX[i]].x;
	}
	out.x/=IX.size();
	out.y/=IX.size();
	out.width=0;
	out.height=0;
	/*out.width=max_x-min_x;
	out.height=out.width;
	out.x-=out.width/2;
	out.y-=out.height/2;*/
	return out;
}

//void GetFPartsRects(vector<Point> &fpoints,vector<Rect> &part_rects)
void PartDetector::GetFPartsRects(vector<Rect> fpoints,vector<Rect> &part_rects)
{
	bool is_mid_rect=true;
	part_rects.push_back(fpoints[0]);
	vector<int> IX;
	IX.clear();
	if (is_mid_rect)
	{
		//Rect leye,reye,nose,mouth;
		

		IX.push_back(19);IX.push_back(20);IX.push_back(21);IX.push_back(22);IX.push_back(23);IX.push_back(24); // left eye
		part_rects.push_back(GetMidRect(fpoints,IX));
		IX.clear();
		IX.push_back(25);IX.push_back(26);IX.push_back(27);IX.push_back(28);IX.push_back(29);IX.push_back(30); // right eye
		part_rects.push_back(GetMidRect(fpoints,IX));
		IX.clear();
		IX.push_back(13);IX.push_back(14);IX.push_back(15);IX.push_back(16);IX.push_back(17);IX.push_back(18); // nose
		part_rects.push_back(GetMidRect(fpoints,IX));

		//IX.clear();
		//IX.push_back(31);IX.push_back(32);IX.push_back(33);IX.push_back(34);IX.push_back(35);IX.push_back(36);  //lips
		//IX.push_back(37);IX.push_back(38);IX.push_back(39);IX.push_back(40);IX.push_back(41);IX.push_back(42);
		//part_rects.push_back(GetMidRect(fpoints,IX));

		//part_rects.push_back(fpoints[19]);part_rects.push_back(fpoints[20]);part_rects.push_back(fpoints[21]);
		//part_rects.push_back(fpoints[22]);part_rects.push_back(fpoints[23]);part_rects.push_back(fpoints[24]); // left eye
		//part_rects.push_back(GetMidRect(fpoints,IX));
		//part_rects.clear();
		//part_rects.push_back(25);part_rects.push_back(26);part_rects.push_back(27);part_rects.push_back(28);part_rects.push_back(29);part_rects.push_back(30); // right eye
		//part_rects.push_back(GetMidRect(fpoints,IX));
		//part_rects.clear();
		//part_rects.push_back(13);part_rects.push_back(14);part_rects.push_back(15);part_rects.push_back(16);part_rects.push_back(17);part_rects.push_back(18); // nose
		//part_rects.push_back(GetMidRect(fpoints,IX));

		//part_rects.clear();
		//part_rects.push_back(31);part_rects.push_back(32);part_rects.push_back(33);part_rects.push_back(34);part_rects.push_back(35);part_rects.push_back(36);  //lips
		//part_rects.push_back(37);part_rects.push_back(38);part_rects.push_back(39);part_rects.push_back(40);part_rects.push_back(41);part_rects.push_back(42);
		//part_rects.push_back(GetMidRect(fpoints,IX));
		
		part_rects.push_back(fpoints[31+1]);
		part_rects.push_back(fpoints[37+1]);

		/*part_rects.push_back(fpoints[4+1]);
		part_rects.push_back(fpoints[5+1]);
		part_rects.push_back(fpoints[10+1]);
		part_rects.push_back(fpoints[12+1]);*/


		//part_rects.push_back(fpoints[40+1]);
		//part_rects.push_back(fpoints[0+1]);
		//part_rects.push_back(fpoints[9+1]);
	}
	else
	{
		part_rects.push_back(fpoints[19]);
		part_rects.push_back(fpoints[22]);
		part_rects.push_back(fpoints[25]);
		part_rects.push_back(fpoints[28]);
		part_rects.push_back(fpoints[13]);
		part_rects.push_back(fpoints[31]);
		part_rects.push_back(fpoints[37]);

	}
	//fpoints[19]
}

int PartDetector::GetCropFace(Mat &img, vector<Rect> &face_rects, Mat &img_out, bool is_rotate)
{
	Size new_sz(128,128);
	const double kBiggerRoi=1.2;
	if (face_rects.size()<1)  
		return -1;
	//Mat img_crop;
	double angle;
	Mat img_rot; //ToDo подумать можно ли не вводить доп переменную или ен вращать все изображение целиком
	if (face_rects.size()>2 && is_rotate)
	{
		int mx,my;
		mx=face_rects[0].x+face_rects[0].width/2;
		my=face_rects[0].y+face_rects[0].height/2;
		angle=std::atan(double(face_rects[2].y-face_rects[1].y)/double(face_rects[2].x-face_rects[1].x));
		angle*=180/3.14;
		//img_crop=img(big_rect);
		//imrotate()
		Mat rot_mat = getRotationMatrix2D(Point(mx,my), angle, 1.0);
		
		if (abs(angle)<30)
			warpAffine(img, img_rot, rot_mat, img.size());
		else 
			img_rot=img;
		//rectangle(img_crop,face_rect[0],Scalar(255,0,0));
		//rectangle(img_crop,big_rect,Scalar(255,0,0));
		//resize(img_crop,img_crop,Size(100,100));
	}
	img_out=img_rot(face_rects[0]);
	double multi_x, multi_y,dx,dy;
	dx=face_rects[0].x;
	dy=face_rects[0].y;
	multi_x=double(new_sz.width)/double(face_rects[0].width);
	multi_y=double(new_sz.height)/double(face_rects[0].height);
	resize(img_out,img_out,new_sz);
	for (int i=0;i<face_rects.size();i++)
	{
		face_rects[i].x-=dx;
		face_rects[i].y-=dy;
		face_rects[i].x*=multi_x;
		face_rects[i].y*=multi_y;
		face_rects[i].width*=multi_x;
		face_rects[i].height*=multi_y;
		if (is_rotate)
		{
			double mx=face_rects[i].x+face_rects[i].width/2-face_rects[0].width/2;
			double my=face_rects[i].y+face_rects[i].height/2-face_rects[0].height/2;
			double alpha=-3.14*angle/180;
			double mx2,my2;
			mx2 = mx*cos(alpha) - my*sin(alpha);
			my2 = mx*sin(alpha) + my*cos(alpha);
			face_rects[i].x=mx2-face_rects[i].width/2+face_rects[0].width/2;
			face_rects[i].y=my2-face_rects[i].height/2+face_rects[0].height/2;
		}
	}
	//imshow("img",img_out);
	//waitKey(0);
	return 0;
}

int  PartDetector::Init(char* cascadeName,char* detectionModel, char* trackingModel)
{
	//CascadeClassifier cascade;
	//imshow("img",img);

	if( !m_cascade.load( cascadeName ))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}
	//INTRAFACE::FaceAlignment *fa;
	//char detectionModel[] = "../models/DetectionModel-v1.5.bin";
	//char trackingModel[] = "../models/TrackingModel-v1.10.bin";
	//INTRAFACE::XXDescriptor xxd(4);
	m_fa = new INTRAFACE::FaceAlignment(detectionModel, trackingModel, &m_xxd);

	if (!m_fa->Initialized()) {
	cerr << "FaceAlignment cannot be initialized." << endl;
	return -2;
	}

}
int PartDetector::ExtractAllFaces(Mat& img,vector<Mat> &img_crop_out, vector<vector<Rect>> &fpoints,vector<vector<Rect>> &face_parts)
{
	vector<vector<Rect>> face_rects;
	//imshow("img",img);
	//waitKey(0);
	if (GetAllFaces( img,face_rects)<0)
		return -1;
	//face_parts=face_rects;
	fpoints.resize(face_rects.size());
	for (int i=0;i<face_rects.size();i++)
	{
		vector<Rect> part_rects;
		Mat img_crop;
		GetFPartsRects(face_rects[i],part_rects);
		face_parts.push_back(part_rects);
		GetCropFace(img,part_rects,img_crop); 
		img_crop_out.push_back(img_crop);
		//double coeff=0.35;
		StandartRectSize(part_rects, rect_sz_coeff,img_crop.size());
		for (int j=1;j<part_rects.size();j++)
			fpoints[i].push_back(part_rects[j]);

	}
	return 0;
}
int PartDetector::ExtractFace(Mat& img,Mat &img_crop_out, vector<Rect> &fpoints)
{
	    vector<Rect> face_rects;
		//imshow("img",img);
		//waitKey(0);
		if (GetFirstFace( img,face_rects)<0)
			return -1;
		vector<Rect> part_rects;   
		GetFPartsRects(face_rects,part_rects);
		GetCropFace(img,part_rects,img_crop_out); 
		//vector<Rect> part_rects2;
		//for (int k=2;k<5;k++)
		{
			//double coeff=0.25;
			StandartRectSize(part_rects, rect_sz_coeff,img_crop_out .size());
			for (int j=1;j<part_rects.size();j++)
				fpoints.push_back(part_rects[j]);
		}
		//for (int j=0;j<fpoints.size();j++)
		//	rectangle(img_crop_out,fpoints[j],Scalar(255,0,0));

		//imshow("parts",img_crop_out);
		//waitKey(0);
}
int PartDetector::ExtractCSVFaces(string dir_full_im, vector<Mat>& imgs, vector<int>& labels_out, vector<vector<Rect>> &fpoints)
{
	vector<int> labels;
	vector<string> im_names;
	read_csv_names(dir_full_im, im_names, labels);
	const double kBiggerRoi=1.2;

	if (!is_init)
		return -1;

	vector<Rect> face_rects; 
	Mat img_crop;
	Mat image;
	for (int i=0;i<im_names.size();i++)
	//for (int i=12;i<14;i++)
	{
		//int64 start = cvGetTickCount();
		image=imread(im_names[i],0);
		if (image.empty())
			continue;
		vector<Rect> part_rects;
		if (ExtractFace(image,img_crop,part_rects)<0)
			continue;
		imgs.push_back(img_crop.clone());
		fpoints.push_back(part_rects);
 		labels_out.push_back(labels[i]);
	}
	

	return 0;
}
void PartDetector::Release()
{
	delete m_fa;
}