#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>
#include <depthai/depthai.hpp>

class BST_CAM
{
private:
    // BST_CAM driver config
    struct Config
    {
        static const int CAMERA_FPS = 20;
        static const int IMU_FPS = 400;

        static const int IMAGE_WIDTH = 640;
        static const int IMAGE_HEIGHT = 400;

        static const int BATCH_REPORT_THRESHOLD = 1;
        static const int MAX_BATCH_REPORTS = 10;
    };

public:
    BST_CAM()
        : Pipeline(),
        USBDevice(),
        CalibrationHandler(),
        FocalX(0), FocalY(0), StereoBaseline(0)
    {
        InitializePipeline();
        InitializeUSB();
    }

    // Holds the latest BST_CAM readings and calibration data
    struct InputData
    {
        cv::Mat Rotation;  // Rotation matrix from IMU
        cv::Mat Left;      // Left camera frame
        cv::Mat Right;     // Right camera frame
        float BaselineDistance;
        float fx;
        float fy;
    };

    // Retrieve and assemble all input data needed by the BST_SLAM engine
    InputData GetInputData()
    {
        InputData data;

        // --- Retrieve frames ---
        auto imuData = IMUQueue->get<dai::IMUData>();
        data.Rotation = ComputeIMURotation(imuData);
        data.Left = LeftQueue->get<dai::ImgFrame>()->getCvFrame();
        data.Right = RightQueue->get<dai::ImgFrame>()->getCvFrame();

        // --- Calibration ---
        data.BaselineDistance = StereoBaseline;
        data.fx = FocalX;
        data.fy = FocalY;

        return data;
    }

private:
    dai::Device USBDevice;       // OAK-D device
    dai::Pipeline Pipeline;      // DepthAI pipeline
    dai::CalibrationHandler CalibrationHandler;

    std::shared_ptr<dai::DataOutputQueue> IMUQueue;
    std::shared_ptr<dai::DataOutputQueue> LeftQueue;
    std::shared_ptr<dai::DataOutputQueue> RightQueue;

    float FocalX;
    float FocalY;
    float StereoBaseline;

    cv::Mat FirstIMURotation; // Used to normalize the first IMU reading

    void InitializeUSB()
    {
        USBDevice.startPipeline(Pipeline);

        // Read calibration
        CalibrationHandler = USBDevice.readCalibration();
        auto intrinsics = CalibrationHandler.getCameraIntrinsics(
            dai::CameraBoardSocket::CAM_B,
            Config::IMAGE_WIDTH,
            Config::IMAGE_HEIGHT
        );

        // Focal lengths in pixel units
        FocalX = intrinsics[0][0];
        FocalY = intrinsics[1][1];

        // Stereo baseline in meters
        StereoBaseline = CalibrationHandler.getBaselineDistance() * 0.01f;

        // Initialize queues
        IMUQueue = USBDevice.getOutputQueue("IMU", 1, false);
        LeftQueue = USBDevice.getOutputQueue("Left", 1, false);
        RightQueue = USBDevice.getOutputQueue("Right", 1, false);
    }

    void InitializePipeline()
    {
        // Create IMU node + link out
        auto imuNode = Pipeline.create<dai::node::IMU>();
        auto imuXLinkOut = Pipeline.create<dai::node::XLinkOut>();
        imuXLinkOut->setStreamName("IMU");
        ConfigureIMU(imuNode);
        imuNode->out.link(imuXLinkOut->input);

        // Create left + right mono cameras
        auto leftMono = Pipeline.create<dai::node::MonoCamera>();
        auto rightMono = Pipeline.create<dai::node::MonoCamera>();

        // Create xlink outs for camera streams
        auto leftXLinkOut = Pipeline.create<dai::node::XLinkOut>();
        auto rightXLinkOut = Pipeline.create<dai::node::XLinkOut>();

        leftXLinkOut->setStreamName("Left");
        rightXLinkOut->setStreamName("Right");

        InitializeCamera(leftMono, dai::CameraBoardSocket::CAM_B);
        InitializeCamera(rightMono, dai::CameraBoardSocket::CAM_C);

        leftMono->out.link(leftXLinkOut->input);
        rightMono->out.link(rightXLinkOut->input);
    }

    void InitializeCamera(std::shared_ptr<dai::node::MonoCamera> camera,
        dai::CameraBoardSocket socket)
    {
        camera->setBoardSocket(socket);
        camera->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
        camera->setFps(Config::CAMERA_FPS);
    }

    void ConfigureIMU(std::shared_ptr<dai::node::IMU> imuNode)
    {
        imuNode->enableIMUSensor(dai::IMUSensor::ROTATION_VECTOR, Config::IMU_FPS);
        imuNode->setBatchReportThreshold(Config::BATCH_REPORT_THRESHOLD);
        imuNode->setMaxBatchReports(Config::MAX_BATCH_REPORTS);
    }

    // Convert DepthAI IMU data to a usable cv::Mat rotation
    cv::Mat ComputeIMURotation(const std::shared_ptr<dai::IMUData>& imuData)
    {
        cv::Mat rotationMat = cv::Mat::eye(3, 3, CV_32F);

        auto& rVec = imuData->packets.back().rotationVector;
        cv::Mat tmp = (cv::Mat)cv::Quatf(rVec.k, -rVec.i, -rVec.j, rVec.real).toRotMat3x3();

        // Save the first rotation so we can "zero" out the IMU orientation
        if (FirstIMURotation.empty()) {
            FirstIMURotation = tmp.clone();
        }

        // Compute rotation relative to the first IMU orientation
        rotationMat = (tmp * FirstIMURotation.inv()).inv();
        rotationMat.convertTo(rotationMat, CV_32F);
        return rotationMat;
    }
};