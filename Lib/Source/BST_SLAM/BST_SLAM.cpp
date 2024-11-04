#include "BST_SLAM.hpp"

SlamFrame::SlamFrame() :
  Left(cv::Mat())
, Right(cv::Mat())
, Rotation(cv::Mat())
, fx(1)
, fy(1)
, Timestamp(clock()) { ; }

SlamFrame SlamFrame::Clone() 
{
	SlamFrame Clone;
	Clone.Left = Left.clone();
	Clone.Right = Right.clone();
	Clone.Rotation = Rotation.clone();
	Clone.fx = fx;
	Clone.fy = fy;
	Clone.Timestamp = Timestamp;
	return Clone;
}

SlamImplementation::SlamImplementation(float Baseline) :
	  StereoBaselineDistance(Baseline)
	, CurrentWorldSpaceSensorPosition(0, 0, 0)
	, PreviousTimestamp(clock())
	, CurrentWorldSpaceSensorDirection(0, 0, 0)
	, IsRotatingTooMuchToCalculateTranslation(true)
{
	OrbFeatureDetector = cv::ORB::create(SlamConfig::MaxFeatures);
	BruteForceFeatureMatcher = cv::BFMatcher::create();
	StereoMatcherSGBM = cv::StereoSGBM::create(0, 64, 7, 392, 1568);
}

cv::Vec3f SlamImplementation::CalculateCurrentWorldSpaceSensorPosition(const cv::Mat& Left, const cv::Mat& Right, const cv::Mat& Rotation, float fx, float fy) 
{
	SlamFrame CurrentSlamFrame;
	CurrentSlamFrame.Left = Left.clone();
	CurrentSlamFrame.Right = Right.clone();
	CurrentSlamFrame.Rotation = Rotation.clone();
	CurrentSlamFrame.fx = fx;
	CurrentSlamFrame.fy = fy;
	CurrentSlamFrame.Timestamp = clock();

	if (!PreviousSlamFrame.Left.empty()) 
	{
		bool WasCalculationSuccessful = false;
		cv::Vec3f CurrentWorldSpaceSensorTranslation = CalculateWorldSpaceSensorTranslation(PreviousSlamFrame, CurrentSlamFrame, 0, WasCalculationSuccessful);

		if (WasCalculationSuccessful) CurrentWorldSpaceSensorPosition += CurrentWorldSpaceSensorTranslation;
	}

	PreviousSlamFrame = CurrentSlamFrame.Clone();
	return CurrentWorldSpaceSensorPosition;
}

cv::Vec3f SlamImplementation::CalculateWorldSpaceSensorTranslation(SlamFrame SlamStartFrame, SlamFrame SlamEndFrame, float MinConfidenceScore, bool& WasCalculationSuccessful) 
{
	WasCalculationSuccessful = false;
	cv::Vec3f WorldSpaceSensorTranslation(0, 0, 0);
	try
	{
		float ImageDownscaler = 256.0f / std::min(SlamEndFrame.Left.rows, SlamEndFrame.Left.cols);
		cv::Mat LeftStartingImage, RightStartingImage, LeftEndingImage, NextRightImg;
		cv::resize(SlamStartFrame.Left, LeftStartingImage, cv::Size(), ImageDownscaler, ImageDownscaler);
		cv::resize(SlamStartFrame.Right, RightStartingImage, cv::Size(), ImageDownscaler, ImageDownscaler);
		cv::resize(SlamEndFrame.Left, LeftEndingImage, cv::Size(), ImageDownscaler, ImageDownscaler);
		cv::resize(SlamEndFrame.Right, NextRightImg, cv::Size(), ImageDownscaler, ImageDownscaler);
		float fx = SlamStartFrame.fx * ImageDownscaler;
		float fy = SlamStartFrame.fy * ImageDownscaler;

		cv::Mat CameraIntrinsics = BST_SLAM::CalculateCameraIntrinsics(fx, fy, LeftEndingImage.size());
		cv::Mat RelativeRotation = (SlamStartFrame.Rotation.inv() * SlamEndFrame.Rotation).inv();
		cv::Mat HomographyRotationOnly = CameraIntrinsics * RelativeRotation * CameraIntrinsics.inv();
		cv::warpPerspective(LeftStartingImage, LeftStartingImage, HomographyRotationOnly, LeftEndingImage.size());
		cv::warpPerspective(RightStartingImage, RightStartingImage, HomographyRotationOnly, LeftEndingImage.size());

		std::vector<cv::KeyPoint> StartingKeypoints, EndingKeypoints;
		cv::Mat StartingDescriptors, EndingDescriptors;
		OrbFeatureDetector->detectAndCompute(LeftStartingImage, cv::noArray(), StartingKeypoints, StartingDescriptors);
		OrbFeatureDetector->detectAndCompute(LeftEndingImage, cv::noArray(), EndingKeypoints, EndingDescriptors);
		std::vector<cv::Point2f> StartingPoints, EndingPoints;
		if (!StartingDescriptors.empty() && !EndingDescriptors.empty())
		{
			std::vector<std::vector<cv::DMatch>> Matches;
			BruteForceFeatureMatcher->knnMatch(StartingDescriptors, EndingDescriptors, Matches, 2);
			for (const auto& Match : Matches)
			{
				if (Match.size() >= 2 && Match[0].distance < SlamConfig::LowesRatio * Match[1].distance)
				{
					StartingPoints.push_back(StartingKeypoints[Match[0].queryIdx].pt);
					EndingPoints.push_back(EndingKeypoints[Match[0].trainIdx].pt);
				}
			}
			if (StartingPoints.size() >= SlamConfig::MinFeatures)
			{
				cv::Mat Inliers;
				cv::findHomography(StartingPoints, EndingPoints, Inliers, cv::USAC_MAGSAC);
				Inliers.convertTo(Inliers, CV_8U);

				std::vector<cv::Point2f> StartingPointsFiltered, EndingPointsFiltered;
				for (int i = 0; i < Inliers.rows; i++)
				{
					if ((int)Inliers.at<uchar>(i) == 1)
					{
						StartingPointsFiltered.push_back(StartingPoints[i]);
						EndingPointsFiltered.push_back(EndingPoints[i]);
					}
				}

				cv::Mat AffinePartial2D = cv::estimateAffinePartial2D(StartingPointsFiltered, EndingPointsFiltered);
				AffinePartial2D.convertTo(AffinePartial2D, CV_32F);
				float CalculatedConfidenceScore = (float)cv::countNonZero(Inliers) / (8.0f + 0.3f * (float)StartingPoints.size());
				if (!AffinePartial2D.empty() && CalculatedConfidenceScore >= MinConfidenceScore)
				{
					cv::vconcat(AffinePartial2D, cv::Vec3f(0, 0, 1).t(), AffinePartial2D);

					cv::Mat DepthMap;
					StereoMatcherSGBM->compute(LeftStartingImage, RightStartingImage, DepthMap);
					DepthMap.convertTo(DepthMap, CV_32F);
					cv::GaussianBlur(DepthMap, DepthMap, cv::Size(21, 21), 1);
					float Depth = StereoBaselineDistance * fx / (std::sqrt(cv::norm(DepthMap)) / SlamConfig::DisparityScale);

					std::vector<cv::Point3f> StartingObjectPoints{ cv::Point3f(0, 0, Depth) };
					std::vector<cv::Point2f> StartingImagePoints;
					std::vector<cv::Point2f> EndingImagePoints;
					cv::projectPoints(StartingObjectPoints, cv::Vec3f(), cv::Vec3f(), CameraIntrinsics, cv::noArray(), StartingImagePoints);
					cv::perspectiveTransform(StartingImagePoints, EndingImagePoints, AffinePartial2D);
					cv::Vec3f Translation
					(
						((EndingImagePoints[0].x - StartingImagePoints[0].x) / fx) * +Depth,
						((EndingImagePoints[0].y - StartingImagePoints[0].y) / fx) * +Depth,
						((AffinePartial2D.at<float>(0, 0) - 1.0) * LeftEndingImage.cols / fx) * -Depth
					);
					if (cv::norm(Translation) > 0) CurrentWorldSpaceSensorDirection = (cv::Vec3f)(cv::Mat)(SlamEndFrame.Rotation * cv::normalize(Translation));
					if (cv::norm(Translation) >= 8) Translation = cv::Vec3f(0, 0, 0);

					float DT = (float)(SlamEndFrame.Timestamp - SlamStartFrame.Timestamp) / (float)CLOCKS_PER_SEC;
					float RelativeRotation = (float)cv::norm(SlamStartFrame.Rotation.inv() * SlamEndFrame.Rotation - cv::Mat::eye(3, 3, CV_32F)) / DT;
					bool IsRotatingTooMuchToCalculateTranslation = RelativeRotation > SlamConfig::trRatio;
					if (!IsRotatingTooMuchToCalculateTranslation)
					{
						WorldSpaceSensorTranslation = CurrentWorldSpaceSensorDirection * cv::norm(Translation);
						WorldSpaceSensorTranslation[0] *= -1;
						WorldSpaceSensorTranslation[2] *= -1;
						WasCalculationSuccessful = true;
					}
					this->IsRotatingTooMuchToCalculateTranslation = IsRotatingTooMuchToCalculateTranslation;
				}
			}
		}
	}
	catch (...) { ; }
	return WorldSpaceSensorTranslation;
}

cv::Mat BST_SLAM::CalculateCameraIntrinsics(float fx, float fy, cv::Size ImageResolution)
{
	cv::Mat CameraIntrinsics = cv::Mat::eye(3, 3, CV_32F);
	CameraIntrinsics.at<float>(0, 0) = fx;
	CameraIntrinsics.at<float>(1, 1) = fy;
	CameraIntrinsics.at<float>(0, 2) = ImageResolution.width / 2.0f;
	CameraIntrinsics.at<float>(1, 2) = ImageResolution.height / 2.0f;
	return CameraIntrinsics;
}

void BST_SLAM::CheckRotation(const cv::Mat& Rotation, float fx, float fy, const cv::Mat& Left, const cv::Mat& Right)
{
	cv::Mat DebugImage;
	cv::addWeighted(Left, 0.5, Right, 0.5, 1, DebugImage);

	cv::Mat CameraIntrinsics = BST_SLAM::CalculateCameraIntrinsics(fx, fy, Left.size());

	cv::warpPerspective(DebugImage, DebugImage, CameraIntrinsics * Rotation * CameraIntrinsics.inv(), Left.size());
	cv::imshow("The rotation should stabilize to zero. Please check.", DebugImage);
}

cv::Vec4f BST_SLAM::ConvertRotation(const cv::Mat& Rotation)
{
	auto Q = cv::Quatf::createFromRotMat(Rotation);
	return cv::Vec4f(-Q.x, +Q.y, -Q.z, +Q.w);
}

cv::Vec3f BST_SLAM::CalculateCurrentWorldSpaceSensorPosition(const cv::Mat& Rotation, float fx, float fy, const cv::Mat& Left, const cv::Mat& Right, const float& StereoBaselineDistance)
{
	static SlamImplementation* Implementation = new SlamImplementation(StereoBaselineDistance);
	return Implementation->CalculateCurrentWorldSpaceSensorPosition(Left, Right, Rotation, fx, fy);
}