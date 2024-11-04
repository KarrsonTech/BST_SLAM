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

#include <depthai/depthai.hpp>

namespace SensorConfig
{
    static int FPS = 20;
    static int ImageWidth = 640;
    static int ImageHeight = 400;
    static int ImuUpdateRate = 400;
    static int ImuBatchReportThreshold = 1;
    static int ImuMaxBatchReports = 10;
};

struct SensorFrame 
{
    float StereoBaselineDistance;
    float fx;
    float fy;

    cv::Mat Left;
    cv::Mat Right;
    cv::Mat Rotation;
};

class OAK_D 
{
    public:
        OAK_D();
        SensorFrame ReadSensorFrame();

    private:
        dai::Device USB;
        void UsbInit();

        dai::Pipeline Pipeline;
        void PipelineInit();

        void CameraInit(std::shared_ptr<dai::node::MonoCamera> Camera, dai::CameraBoardSocket Socket);
        std::shared_ptr<dai::DataOutputQueue> LeftQueue;
        std::shared_ptr<dai::DataOutputQueue> RightQueue;
        dai::CalibrationHandler CalibrationHandler;
        float fx;
        float fy;

        void ImuInit(std::shared_ptr<dai::node::IMU> IMU);
        std::shared_ptr<dai::DataOutputQueue> ImuQueue;
        cv::Mat RotationInit;
        cv::Mat ReadRotation(const std::shared_ptr<dai::IMUData>& IMU);
        static cv::Mat ConvertRotation(float x, float y, float z, float w);
};