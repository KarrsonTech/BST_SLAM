#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>

namespace SlamConfig
{
    static int MinFeatures = 15;
    static int MaxFeatures = 640;
    static float LowesRatio = 0.84375f;
    static float trRatio = 2.0f;
    static int MinDisparity = 16;
    static int NumDisparities = 16;
    static int BlockWindowSize = 7;
    static int P1 = 1024;
    static int P2 = 4096;
    static int Disp12DiffMax = 1;
    static int PreFilterCap = 63;
    static int DisparityRatio = 24;
    static int SpeckleWindowSize = 7;
    static int SpeckleRange = 3;
    static float DisparityScale = 16.0f;
};

struct SlamFrame 
{
    float fx, fy;
    cv::Mat Left;
    cv::Mat Right;
    cv::Mat Rotation;
    clock_t Timestamp;

    SlamFrame();
    SlamFrame Clone();
};

class SlamImplementation 
{
    public:
        SlamImplementation(float Baseline);

        cv::Vec3f CalculateCurrentWorldSpaceSensorPosition(const cv::Mat& Left, const cv::Mat& Right, const cv::Mat& Rotation, float fx, float fy);

    private:
        float StereoBaselineDistance;

        cv::Vec3f CurrentWorldSpaceSensorPosition;
        cv::Vec3f CurrentWorldSpaceSensorDirection;
        bool IsRotatingTooMuchToCalculateTranslation;
        SlamFrame PreviousSlamFrame;
        clock_t PreviousTimestamp;

        cv::Ptr<cv::ORB> OrbFeatureDetector;
        cv::Ptr<cv::BFMatcher> BruteForceFeatureMatcher;
        cv::Ptr<cv::StereoSGBM> StereoMatcherSGBM;
        cv::Vec3f CalculateWorldSpaceSensorTranslation(SlamFrame SlamStartFrame, SlamFrame SlamEndFrame, float MinConfidenceScore, bool& WasCalculationSuccessful);
};

namespace BST_SLAM
{
    cv::Mat CalculateCameraIntrinsics(float fx, float fy, cv::Size ImageResolution);
    void CheckRotation(const cv::Mat& Rotation, float fx, float fy, const cv::Mat& Left, const cv::Mat& Right);
    cv::Vec4f ConvertRotation(const cv::Mat& Rotation);
    cv::Vec3f CalculateCurrentWorldSpaceSensorPosition(const cv::Mat& Rotation, float fx, float fy, const cv::Mat& Left, const cv::Mat& Right, const float& StereoBaselineDistance);
}