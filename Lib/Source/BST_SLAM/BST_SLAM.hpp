#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>

static constexpr double MotionModA    = 100.0;
static constexpr double MotionModB    = 15.24;
static constexpr double MotionModC    = 1.250;
static constexpr double ImageSize     = 366.0;
static constexpr double MinConfidence = .3660;
static constexpr double MaxFeatures   = 666.0;
static constexpr double LowesRatio    = .8660;
static constexpr double CenterPoint   = 183.0;

static void VerifyRotation(const cv::Mat& Rotation, const cv::Mat& Image) {
    cv::Mat DebugImage;
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = K.at<double>(1, 1) = ImageSize;
    K.at<double>(0, 2) = K.at<double>(1, 2) = CenterPoint;
    cv::Mat R = Rotation.clone();
    R.convertTo(R, CV_64F);
    cv::warpPerspective(Image, DebugImage, K * R * K.inv(), Image.size());
    cv::imshow("The rotation should stabilize to zero. Please check.", DebugImage);
}

static cv::Vec4f QuaternionFromRotation(const cv::Mat& Rotation) {
    auto Quaternion = cv::Quatf::createFromRotMat(Rotation);
    return cv::Vec4f(-Quaternion.x, +Quaternion.y, -Quaternion.z, +Quaternion.w);
}

static cv::Vec3f EstimatePosition3D(const cv::Mat& CurrentImage, const cv::Mat& CurrentRotation) {
    static cv::Vec3f WorldPosition(0, 0, 0);
    static cv::Vec3f WorldDirection(0, 0, 0);
    static cv::Mat PreviousImage;
    static cv::Mat PreviousRotation;
    static clock_t PreviousTimestamp = clock();
    static cv::Ptr<cv::ORB> OrbDetector = cv::ORB::create(MaxFeatures);
    static cv::Ptr<cv::BFMatcher> BfMatcher = cv::BFMatcher::create();

    if (!PreviousImage.empty()) {
        cv::Mat ResizedPrevImage, ResizedCurrentImage;
        cv::resize(PreviousImage, ResizedPrevImage, cv::Size(ImageSize, ImageSize), 0, 0, cv::INTER_AREA);
        cv::resize(CurrentImage, ResizedCurrentImage, cv::Size(ImageSize, ImageSize), 0, 0, cv::INTER_AREA);

        cv::Mat CurrentRotationFloat = CurrentRotation;
        CurrentRotationFloat.convertTo(CurrentRotationFloat, CV_32F);
        cv::Mat RotationDelta = (PreviousRotation.inv() * CurrentRotation).inv();
        RotationDelta.convertTo(RotationDelta, CV_64F);

        auto QuaternionRotation = cv::Quatf::createFromRotMat(CurrentRotationFloat);

        cv::Mat CameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        CameraMatrix.at<double>(0, 0) = CameraMatrix.at<double>(1, 1) = ImageSize;
        CameraMatrix.at<double>(0, 2) = CameraMatrix.at<double>(1, 2) = CenterPoint;

        cv::Mat WarpedRotation = CameraMatrix * RotationDelta * CameraMatrix.inv();

        cv::Mat HomographyMatrix = cv::Mat::eye(3, 3, CV_64F);
        try {
            cv::warpPerspective(ResizedPrevImage, ResizedPrevImage, WarpedRotation, ResizedCurrentImage.size());

            std::vector<cv::KeyPoint> PrevKeypoints, CurrentKeypoints;
            cv::Mat PrevDescriptors, CurrentDescriptors;
            OrbDetector->detectAndCompute(ResizedPrevImage, cv::noArray(), PrevKeypoints, PrevDescriptors);
            OrbDetector->detectAndCompute(ResizedCurrentImage, cv::noArray(), CurrentKeypoints, CurrentDescriptors);

            if (PrevKeypoints.size() >= 4 && CurrentKeypoints.size() >= 4) {
                std::vector<cv::Point2f> MatchedPrevPoints, MatchedCurrentPoints;
                std::vector<std::vector<cv::DMatch>> Matches;
                BfMatcher->knnMatch(PrevDescriptors, CurrentDescriptors, Matches, 2);

                for (const auto& Match : Matches) {
                    if (Match.size() >= 2 && Match[0].distance <= LowesRatio * Match[1].distance) {
                        MatchedPrevPoints.push_back(PrevKeypoints[Match[0].queryIdx].pt);
                        MatchedCurrentPoints.push_back(CurrentKeypoints[Match[0].trainIdx].pt);
                    }
                }

                if (MatchedPrevPoints.size() >= 4) {
                    cv::Mat InlierMask;
                    cv::findHomography(MatchedPrevPoints, MatchedCurrentPoints, InlierMask,
                        MatchedPrevPoints.size() >= 15 ? cv::USAC_MAGSAC : cv::RANSAC);

                    int TotalMatches = MatchedPrevPoints.size();

                    for (int i = InlierMask.rows - 1; i >= 0; i--) {
                        if (InlierMask.at<uchar>(i) == 0) {
                            MatchedPrevPoints.erase(MatchedPrevPoints.begin() + i);
                            MatchedCurrentPoints.erase(MatchedCurrentPoints.begin() + i);
                        }
                    }

                    if (MatchedPrevPoints.size() >= 3) {
                        HomographyMatrix = cv::estimateAffinePartial2D(MatchedPrevPoints, MatchedCurrentPoints);
                        float Confidence = static_cast<float>(MatchedPrevPoints.size()) / (8.0f + 0.3f * TotalMatches);

                        if (!HomographyMatrix.empty() && Confidence >= MinConfidence) {
                            cv::vconcat(HomographyMatrix, cv::Vec3d(0, 0, 1).t(), HomographyMatrix);
                        }
                    }
                }
            }
        }
        catch (...) { ; }

        cv::Mat TransformMatrix = cv::Mat::eye(3, 3, CV_64F);
        cv::Point2f CenterPointVec(CenterPoint, CenterPoint);
        std::vector<cv::Point2f> SrcPoints{ CenterPointVec };
        std::vector<cv::Point2f> DstPoints;

        cv::perspectiveTransform(SrcPoints, DstPoints, HomographyMatrix);
        double Scale = HomographyMatrix.at<double>(0, 0);

        TransformMatrix.at<double>(0, 0) = TransformMatrix.at<double>(1, 1) = Scale;
        TransformMatrix.at<double>(0, 2) = DstPoints[0].x - CenterPointVec.x - (Scale - 1.0) * CenterPointVec.x;
        TransformMatrix.at<double>(1, 2) = DstPoints[0].y - CenterPointVec.y - (Scale - 1.0) * CenterPointVec.y;

        static const std::vector<cv::Point3d> ObjectPoints = {
            cv::Point3d(-0.5, 0.5, 0), cv::Point3d(0.5, 0.5, 0),
            cv::Point3d(0.5, -0.5, 0), cv::Point3d(-0.5, -0.5, 0)
        };
        static const std::vector<cv::Point2d> ImagePoints = {
            cv::Point2d(0, 0), cv::Point2d(ImageSize, 0),
            cv::Point2d(ImageSize, ImageSize), cv::Point2d(0, ImageSize)
        };

        std::vector<cv::Point2d> TransformedPoints;
        cv::Mat InitialTransform = TransformMatrix.clone();
        cv::perspectiveTransform(ImagePoints, TransformedPoints, TransformMatrix);

        cv::Vec3d Rvec1, Tvec1, Rvec2, Tvec2;
        cv::solvePnP(ObjectPoints, ImagePoints, CameraMatrix, cv::Mat(), Rvec1, Tvec1);
        cv::solvePnP(ObjectPoints, TransformedPoints, CameraMatrix, cv::Mat(), Rvec2, Tvec2);

        Tvec2 -= Tvec1;
        Tvec2[0] *= -1;
        Tvec2[2] *= -1;

        if (cv::norm(Tvec2) > 0) {
            Tvec2 /= cv::norm(Tvec2);
            WorldDirection = Tvec2;
        }

        clock_t CurrentTimestamp = clock();
        double TimestampDelta = ((double)(CurrentTimestamp - PreviousTimestamp) / (double)CLOCKS_PER_SEC);

        cv::Vec3d PositionDelta = WorldDirection * TimestampDelta * (std::min((cv::norm(CameraMatrix.inv() * TransformMatrix * 
                CameraMatrix - cv::Mat::eye(3, 3, CV_64F)) / TimestampDelta) * MotionModA, MotionModB)) * MotionModC;

        cv::Vec3f WorldSpaceDelta = (cv::Vec3f)(cv::Mat)(CurrentRotationFloat *
            cv::Vec3f(PositionDelta[0], -PositionDelta[1], PositionDelta[2]));
        WorldSpaceDelta[1] *= -1;
        WorldSpaceDelta /= 100.0;

        WorldPosition += WorldSpaceDelta;
    }

    PreviousImage = CurrentImage.clone();
    PreviousRotation = CurrentRotation.clone();
    PreviousTimestamp = clock();

    return WorldPosition;
}