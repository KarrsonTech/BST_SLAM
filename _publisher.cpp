#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat CameraMatrix(float fx, float fy, Size sz);

void CheckSubscriber(Mat R, float fx, float fy, Mat Left, Mat Right) {
	Mat Debug;
	addWeighted(Left, 0.5, Right, 0.5, 1, Debug);
	Mat A = CameraMatrix(fx, fy, Left.size());
	warpPerspective(Debug, Debug, A * R * A.inv(), Left.size());
	imshow("The rotation should stabalize to zero. Please check.", Debug);
}

Mat PrevR;
Mat PrevLeft;
Mat PrevRight;
auto CvStitcher = Stitcher::create(Stitcher::SCANS);

void SetPrev(Mat R, Mat Left, Mat Right, bool Init=false) {
	if (!Init || PrevR.empty()) {
		PrevR = R.clone();
		PrevLeft = Left.clone();
		PrevRight = Right.clone();
	}
}

Vec3f Pos;

Vec3f Tracker(Mat R, float fx, float fy, Mat Left, Mat Right, float Baseline) {
	SetPrev(R, Left, Right, true);
	Left = Left.clone();
	Right = Right.clone();
	float fxy = 180.0 / fminf(Left.rows, Left.cols);
	resize(Left, Left, Size(), fxy, fxy);
	resize(Right, Right, Left.size());
	fx *= fxy;
	fy *= fxy;
	Mat A = CameraMatrix(fx, fy, Left.size());
	warpPerspective(PrevLeft, PrevLeft, (A * (R * PrevR.inv()) * A.inv()).inv(), Left.size());
	warpPerspective(PrevRight, PrevRight, (A * (R * PrevR.inv()) * A.inv()).inv(), Left.size());
	CvStitcher->setPanoConfidenceThresh(0);
	try { CvStitcher->estimateTransform(vector{ PrevLeft, Left }); } catch (...) { ; }
	if (CvStitcher->cameras().size() == 2) {
		Mat T2 = CvStitcher->cameras()[1].R * CvStitcher->cameras()[0].R.inv();
		T2.convertTo(T2, CV_32F);
		Mat Disp;
		static auto CvStereoSGBM = StereoSGBM::create(16, 16, 7, 1176, 4704, 1, 63, 20, 7, 3, StereoSGBM::MODE_SGBM);
		CvStereoSGBM->compute(Left, Right, Disp);
		Disp.convertTo(Disp, CV_32F);
		GaussianBlur(Disp, Disp, Size(21, 21), 3, 3);
		float Depth = Baseline * A.at<float>(0, 0) / (Disp.at<float>(Left.rows / 2.0, Left.cols / 2.0) / 16.0);
		float S = 100;
		vector<Point3f> _3D{
			Point3f(-S, -S, Depth), Point3f(+1, -1, Depth),
			Point3f(-S, +S, Depth), Point3f(+1, +1, Depth)
		};
		vector<Point2f> _2D;
		projectPoints(_3D, Vec3f(), Vec3f(), A, noArray(), _2D);
		perspectiveTransform(_2D, _2D, T2);
		Vec3f _, T;
		solvePnP(_3D, _2D, A, noArray(), _, T, false, SOLVEPNP_AP3P);
		T[0] *= -1;
		T[2] *= -1;
		T = (Mat)(R * T);
		if (norm(T) < 10) {
			Pos += T;
		}
	}
	CvStitcher = Stitcher::create(Stitcher::SCANS);
	SetPrev(R, Left, Right);
	return Pos;
}

Mat CameraMatrix(float fx, float fy, Size sz) {
	Mat A = Mat::eye(3, 3, CV_32F);
	A.at<float>(0, 0) = fx;
	A.at<float>(1, 1) = fy;
	A.at<float>(0, 2) = sz.width / 2.0;
	A.at<float>(1, 2) = sz.height / 2.0;
	return A;
}