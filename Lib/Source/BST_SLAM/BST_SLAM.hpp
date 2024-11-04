#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>
#include "BST_VIO.hpp"

namespace BST_SLAM {
    static void VerifyRotation(cv::Mat Rotation, cv::Mat Image) {
		BST_VIO::VerifyRotation(Rotation, Image);
    }

    static cv::Vec4f QuaternionFromRotation(const cv::Mat& Rotation) {
		return BST_VIO::QuaternionFromRotation(Rotation);
    }

	static cv::Vec3f EstimatePosition3D(cv::Mat CurrentFrame, cv::Mat CurrentRotation) {
		static cv::Vec3f FinalPosition(0, 0, 0);

		// Visual Odometry
		cv::Vec3f TranslationVIO = BST_VIO::EstimateTranslation3D(CurrentFrame, CurrentRotation);
		FinalPosition += TranslationVIO;

		// Loop Closure
		// TODO: Fix FinalPosition using loop closure!

		return FinalPosition;
	}
}