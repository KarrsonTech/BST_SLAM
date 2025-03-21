#pragma once

#include <depthai/depthai.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/opencv.hpp>

class SensorDriver {
public:
    SensorDriver() {
        if (dai::Device::getAllAvailableDevices().size() <= 0) throw std::runtime_error("Couldn't find any available devices.");
        Pipeline = dai::Pipeline();
        Calibration = dai::CalibrationHandler();
        fx = 0;
        fy = 0;
        Baseline = 0; 
        InitializePipeline();
        InitializeDevice();
    }

    struct InputData {
        double fx;
        double fy;
        double Baseline;
        cv::Mat Left;
        cv::Mat Right;
        cv::Mat Rot;
    };

    InputData GetInputData() {
        InputData Data;

        Data.Left = LeftQueue->get<dai::ImgFrame>()->getCvFrame();
        Data.Right = RightQueue->get<dai::ImgFrame>()->getCvFrame();

        Data.Baseline = Baseline;
        Data.fx = fx;
        Data.fy = fy;

        auto IMU = IMUQueue->get<dai::IMUData>();
        Data.Rot = SolveRotation(IMU);

        return Data;
    }

private:
    dai::Pipeline Pipeline;
    dai::CalibrationHandler Calibration;

    std::shared_ptr<dai::DataOutputQueue> IMUQueue;
    std::shared_ptr<dai::DataOutputQueue> LeftQueue;
    std::shared_ptr<dai::DataOutputQueue> RightQueue;

    double fx;
    double fy;
    double Baseline;

    void InitializeDevice() {
        static auto Device = new dai::Device(Pipeline, dai::UsbSpeed::SUPER_PLUS);

        Calibration = Device->readCalibration();
        auto intrinsics = Calibration.getCameraIntrinsics(
            dai::CameraBoardSocket::CAM_B, 640, 400);

        fx = intrinsics[0][0];
        fy = intrinsics[1][1];

        Baseline = Calibration.getBaselineDistance() * 0.01f;

        LeftQueue = Device->getOutputQueue("Left", 1, false);
        RightQueue = Device->getOutputQueue("Right", 1, false);
        IMUQueue = Device->getOutputQueue("IMU", 1, false);
    }

    std::shared_ptr<dai::node::MonoCamera> LeftMono;
    std::shared_ptr<dai::node::MonoCamera> RightMono;

    void InitializePipeline() {
        LeftMono = Pipeline.create<dai::node::MonoCamera>();
        RightMono = Pipeline.create<dai::node::MonoCamera>();

        auto LeftXLink = Pipeline.create<dai::node::XLinkOut>();
        auto RightXLink = Pipeline.create<dai::node::XLinkOut>();

        LeftXLink->setStreamName("Left");
        RightXLink->setStreamName("Right");

        ConfigureCamera(LeftMono, dai::CameraBoardSocket::CAM_B);
        ConfigureCamera(RightMono, dai::CameraBoardSocket::CAM_C);

        LeftMono->out.link(LeftXLink->input);
        RightMono->out.link(RightXLink->input);

        auto IMU = Pipeline.create<dai::node::IMU>();
        auto XLink = Pipeline.create<dai::node::XLinkOut>();
        XLink->setStreamName("IMU");
        ConfigureIMU(IMU);
        IMU->out.link(XLink->input);
    }

    void ConfigureCamera(std::shared_ptr<dai::node::MonoCamera> Camera, dai::CameraBoardSocket Socket) {
        Camera->setBoardSocket(Socket);
        Camera->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
        Camera->setFps(25);
    }

    void ConfigureIMU(std::shared_ptr<dai::node::IMU> IMU) {
        IMU->enableIMUSensor(dai::IMUSensor::ARVR_STABILIZED_ROTATION_VECTOR, 100);
        IMU->setMaxBatchReports(1);
        IMU->setBatchReportThreshold(2);
    }

    cv::Mat SolveRotation(std::shared_ptr<dai::IMUData> IMU) {
        auto& _R = IMU->packets.back().rotationVector;
        cv::Mat R = (cv::Mat)(cv::Quatd(_R.real, -_R.j, -_R.k, _R.i) *
        cv::Quatd::createFromEulerAngles(cv::Vec3d((90.0 / 180.0) * 
        CV_PI, 0, 0), cv::QuatEnum::EXT_XYZ)).toRotMat3x3();
        R.convertTo(R, CV_64F);
        cv::Mat YawMat = R.clone();
        YawMat.convertTo(YawMat, CV_64F);
        double YawVal = atan2(YawMat.at<double>(0, 2), YawMat.at<double>(0, 0));
        YawMat = cv::Mat::zeros(3, 3, CV_64F);
        YawMat.at<double>(0, 0) = cos(YawVal);
        YawMat.at<double>(0, 2) = sin(YawVal);
        YawMat.at<double>(1, 1) = 1.0;
        YawMat.at<double>(2, 0) = -sin(YawVal);
        YawMat.at<double>(2, 2) = cos(YawVal);
        YawMat.convertTo(YawMat, CV_64F);
        static cv::Mat FirstYawMat = YawMat.clone();
        R = FirstYawMat.inv() * R;
        R.convertTo(R, CV_64F);
        return R;
    }
};