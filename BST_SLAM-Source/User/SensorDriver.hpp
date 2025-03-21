#pragma once

#include <depthai/depthai.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/opencv.hpp>

class SensorDriver {
public:
    SensorDriver() : Pipeline(), Device(), Calibration(), fx(0), fy(0), Baseline(0) {
        InitializePipeline();
        InitializeDevice();
    }

    struct InputData {
        float fx;
        float fy;
        float Baseline;
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
    std::shared_ptr<dai::Device> Device;
    dai::Pipeline Pipeline;
    dai::CalibrationHandler Calibration;

    std::shared_ptr<dai::DataOutputQueue> IMUQueue;
    std::shared_ptr<dai::DataOutputQueue> LeftQueue;
    std::shared_ptr<dai::DataOutputQueue> RightQueue;

    float fx;
    float fy;
    float Baseline;

    void InitializeDevice() {
        Device = std::make_shared<dai::Device>(Pipeline, dai::UsbSpeed::SUPER_PLUS);

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
        IMU->enableIMUSensor(dai::IMUSensor::GAME_ROTATION_VECTOR, 125);
        IMU->setMaxBatchReports(1);
        IMU->setBatchReportThreshold(2);
    }

    cv::Mat SolveRotation(std::shared_ptr<dai::IMUData> IMU) {
        auto& _R = IMU->packets.back().rotationVector;
        cv::Mat R = (cv::Mat)(cv::Quatf(_R.real, -_R.j, -_R.k, _R.i) *
            cv::Quatf::createFromEulerAngles(cv::Vec3f((90.0f / 180.0f) * 
                CV_PI, 0, 0), cv::QuatEnum::EXT_XYZ)).toRotMat3x3();
        R.convertTo(R, CV_32F);
        return R;
    }
};