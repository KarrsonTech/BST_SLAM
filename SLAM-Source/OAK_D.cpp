#include "OAK_D.hpp"

OAK_D::OAK_D()
    : USB()
    , Pipeline()
    , CalibrationHandler()
    , fx(1.0f)
    , fy(1.0f) 
{
    PipelineInit();
    UsbInit();
}

SensorFrame OAK_D::ReadSensorFrame() 
{
    SensorFrame Frame;
    Frame.Left = LeftQueue->get<dai::ImgFrame>()->getCvFrame();
    Frame.Right = RightQueue->get<dai::ImgFrame>()->getCvFrame();
    auto IMU = ImuQueue->get<dai::IMUData>();
    Frame.Rotation = ReadRotation(IMU);
    Frame.Acceleration = ReadAcceleration(IMU);
    Frame.StereoBaselineDistance = CalibrationHandler.getBaselineDistance();
    Frame.fx = fx;
    Frame.fy = fy;
    return Frame;
}

void OAK_D::UsbInit() 
{
    USB.startPipeline(Pipeline);
    CalibrationHandler = USB.readCalibration();
    auto CameraIntrinsics = CalibrationHandler.getCameraIntrinsics(dai::CameraBoardSocket::CAM_B, SensorConfig::ImageWidth, SensorConfig::ImageHeight);
    fx = CameraIntrinsics[0][0];
    fy = CameraIntrinsics[1][1];

    LeftQueue = USB.getOutputQueue("Left", 1, false);
    RightQueue = USB.getOutputQueue("Right", 1, false);
    ImuQueue = USB.getOutputQueue("IMU", 1, false);
}

void OAK_D::PipelineInit() 
{
    auto _Left = Pipeline.create<dai::node::MonoCamera>();
    auto _Right = Pipeline.create<dai::node::MonoCamera>();
    auto _IMU = Pipeline.create<dai::node::IMU>();

    auto Left = Pipeline.create<dai::node::XLinkOut>();
    auto Right = Pipeline.create<dai::node::XLinkOut>();
    auto IMU = Pipeline.create<dai::node::XLinkOut>();

    Left->setStreamName("Left");
    Right->setStreamName("Right");
    IMU->setStreamName("IMU");

    CameraInit(_Left, dai::CameraBoardSocket::CAM_B);
    CameraInit(_Right, dai::CameraBoardSocket::CAM_C);
    ImuInit(_IMU);

    _Left->out.link(Left->input);
    _Right->out.link(Right->input);
    _IMU->out.link(IMU->input);
}

void OAK_D::CameraInit(std::shared_ptr<dai::node::MonoCamera> Camera, dai::CameraBoardSocket Socket)
{
    Camera->setBoardSocket(Socket);
    Camera->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    Camera->setFps((float)SensorConfig::FPS);
}

void OAK_D::ImuInit(std::shared_ptr<dai::node::IMU> IMU) 
{
    IMU->enableIMUSensor(dai::IMUSensor::ROTATION_VECTOR, SensorConfig::ImuUpdateRate);
    IMU->setBatchReportThreshold(SensorConfig::ImuBatchReportThreshold);
    IMU->setMaxBatchReports(SensorConfig::ImuMaxBatchReports);
}

cv::Mat OAK_D::ReadRotation(const std::shared_ptr<dai::IMUData>& IMU)
{
    auto Q = IMU->packets.back().rotationVector;
    cv::Mat _Rotation = ConvertRotation(-Q.i, -Q.j, +Q.real, +Q.k);
    if (RotationInit.empty()) RotationInit = _Rotation.clone();
    return (_Rotation * RotationInit.inv()).inv();
}

cv::Vec3f OAK_D::ReadAcceleration(const std::shared_ptr<dai::IMUData>& IMU)
{
    auto A = IMU->packets.back().acceleroMeter;
    return cv::Vec3f(+A.x, +A.y, -A.z);
}

cv::Mat OAK_D::ConvertRotation(float x, float y, float z, float w) 
{
    return (cv::Mat)cv::Quatf(w, x, y, z).toRotMat3x3();
}