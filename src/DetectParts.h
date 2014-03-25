#include <vector>
#include <opencv2/core/core.hpp>
using namespace cv;
#define rect_sz_coeff 0.35
class PartDetector {
	//static const float rect_sz_coeff=0.35;
	string cascadeName;
	cv::Mat m_FacePoints;
	bool is_init;
	INTRAFACE::FaceAlignment *m_fa;
	INTRAFACE::XXDescriptor m_xxd;
	CascadeClassifier m_cascade;
public:
	
	int ExtractFace(Mat& img,Mat &img_crop_out, vector<Rect> &fpoints);
	int ExtractAllFaces(Mat& img,vector<Mat> &img_crop_out, vector<vector<Rect>> &fpoints,vector<vector<Rect>> &face_parts);
	int ExtractCSVFaces(string dir_full_im, vector<Mat>& imgs, vector<int>& labels_out, vector<vector<Rect>> &fpoints);
	//int Init(string cascadeName,string detectionModel, string trackingModel);
	int Init(char* cascadeName,char* detectionModel, char* trackingModel);
private:
	void Release();
	int GetCropFace(Mat &img, vector<Rect> &face_rects, Mat &img_out, bool is_rotate=true);
	void GetFPartsRects(vector<Rect> fpoints,vector<Rect> &part_rects);
	Rect GetMidRect(vector<Rect> &fpoints,vector<int> IX);
	void StandartRectSize(vector<Rect> &rects,double scale_face_part, Size im_sz, bool is_lbph=false);
	Rect IncreaseRectSize(Rect src,double coeff, Size im_sz);
	void read_csv_names(const string& filename, vector<string>& im_names, vector<int>& labels, char separator = ';');
	void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';');
	//int GetFirstFace( Mat& frame, CascadeClassifier& face_cascade, vector<Rect> &out_rect, INTRAFACE::FaceAlignment *fa);
	int GetFirstFace( Mat& frame, vector<Rect> &out_rect);
	int GetAllFaces( Mat& frame, vector<vector<Rect>> &out_rect);
};