#include "OAKCamera.h"

#define TEST_MODE_CHECK_INPUT_DATA	'I'	// START HERE
#define TEST_MODE_CHECK_OUTPUT_DATA	'O'	// END HERE
// SELECT BELOW
#define TEST_MODE TEST_MODE_CHECK_INPUT_DATA

void CheckSubscriber(cv::Mat R, float fx, float fy, cv::Mat Left, cv::Mat Right);
cv::Vec3f Tracker(cv::Mat R, float fx, float fy, cv::Mat Left, cv::Mat Right, float Baseline);
cv::Vec4d Rot(cv::Mat R);

int main() {
	OAKCamera camera = OAKCamera();
	while (true) {
		auto frame = camera.getFrame();
		cv::Mat R = frame.rotation;
		float fx = frame.fx;
		float fy = frame.fy;
		cv::Mat Left = frame.left;
		cv::Mat Right = frame.right;
		float Baseline = frame.baseline; // MUST BE IN CENTIMETERS!
		if (TEST_MODE == TEST_MODE_CHECK_INPUT_DATA) {
			CheckSubscriber(R, fx, fy, Left, Right);
			cv::waitKey(1);
		}
		else if (TEST_MODE == TEST_MODE_CHECK_OUTPUT_DATA) {
			cv::Vec3f Pos = Tracker(R, fx, fy, Left, Right, Baseline);
			std::cout << Pos << Rot(R) << std::endl;
		}
	}
}

#include <opencv2/core/quaternion.hpp>
cv::Vec4d Rot(cv::Mat R) {
	auto Q = cv::Quatf::createFromRotMat(R);
	return cv::Vec4d(-Q.x, Q.y, -Q.z, Q.w);
}
