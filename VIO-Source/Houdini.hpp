#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>
namespace cv::detail::Houdini {
	cv::Vec3f tvec(const cv::Mat& Img, const cv::Mat& Rot);
}
