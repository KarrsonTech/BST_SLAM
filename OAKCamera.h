#pragma once
#include <depthai/depthai.hpp>
#include <opencv2/opencv.hpp>

class OAKCamera {
public:
    struct Frame {
        cv::Mat left;
        cv::Mat right;
        cv::Mat rotation;
        float baseline;
        float fx;
        float fy;
    };

    OAKCamera(int fps = 20) {
        setupPipeline(fps);
        device.startPipeline(pipeline);
        calib = device.readCalibration();
        
        // Get intrinsics once during initialization
        auto intrinsics = calib.getCameraIntrinsics(dai::CameraBoardSocket::CAM_B, 640, 400);
        fx = intrinsics[0][0];
        fy = intrinsics[1][1];

        // Create output queues
        leftQueue = device.getOutputQueue("left", 1, false);
        rightQueue = device.getOutputQueue("right", 1, false);
        imuQueue = device.getOutputQueue("imu", 1, false);
    }

    Frame getFrame() {
        Frame frame;
        
        // Get latest frames
        frame.left = leftQueue->get<dai::ImgFrame>()->getCvFrame();
        frame.right = rightQueue->get<dai::ImgFrame>()->getCvFrame();
        
        // Get IMU rotation
        auto imu = imuQueue->get<dai::IMUData>();
        frame.rotation = getRotationMatrix(imu);
        
        // Set calibration data
        frame.baseline = calib.getBaselineDistance();
        frame.fx = fx;
        frame.fy = fy;
        
        return frame;
    }

private:
    dai::Device device;
    dai::Pipeline pipeline;
    dai::CalibrationHandler calib;
    cv::Mat initial_rotation;
    float fx, fy;
    
    std::shared_ptr<dai::DataOutputQueue> leftQueue;
    std::shared_ptr<dai::DataOutputQueue> rightQueue;
    std::shared_ptr<dai::DataOutputQueue> imuQueue;

    cv::Mat getRotationMatrix(const std::shared_ptr<dai::IMUData>& imu) {
        auto quat = imu->packets.back().rotationVector;
        cv::Mat current_R = quaternionToRotationMatrix(
            +imu->packets.back().rotationVector.k,
            -imu->packets.back().rotationVector.i,
            -imu->packets.back().rotationVector.j,
            +imu->packets.back().rotationVector.real
        );
        
        if (initial_rotation.empty()) {
            initial_rotation = current_R.clone();
        }
        
        return (current_R * initial_rotation.inv()).inv();
    }

    cv::Mat quaternionToRotationMatrix(float w, float x, float y, float z) {
        cv::Mat R = cv::Mat::zeros(3, 3, CV_32F);
        
        R.at<float>(0,0) = 1 - 2*y*y - 2*z*z;
        R.at<float>(0,1) = 2*x*y - 2*w*z;
        R.at<float>(0,2) = 2*x*z + 2*w*y;
        
        R.at<float>(1,0) = 2*x*y + 2*w*z;
        R.at<float>(1,1) = 1 - 2*x*x - 2*z*z;
        R.at<float>(1,2) = 2*y*z - 2*w*x;
        
        R.at<float>(2,0) = 2*x*z - 2*w*y;
        R.at<float>(2,1) = 2*y*z + 2*w*x;
        R.at<float>(2,2) = 1 - 2*x*x - 2*y*y;
        
        return R;
    }

    void setupPipeline(int fps) {
        auto left = pipeline.create<dai::node::MonoCamera>();
        auto right = pipeline.create<dai::node::MonoCamera>();
        auto imu = pipeline.create<dai::node::IMU>();
        
        auto leftOut = pipeline.create<dai::node::XLinkOut>();
        auto rightOut = pipeline.create<dai::node::XLinkOut>();
        auto imuOut = pipeline.create<dai::node::XLinkOut>();

        leftOut->setStreamName("left");
        rightOut->setStreamName("right");
        imuOut->setStreamName("imu");

        // Configure cameras
        configureCam(left, dai::CameraBoardSocket::CAM_B, fps);
        configureCam(right, dai::CameraBoardSocket::CAM_C, fps);

        // Configure IMU
        imu->enableIMUSensor(dai::IMUSensor::ROTATION_VECTOR, 400);
        imu->setBatchReportThreshold(1);
        imu->setMaxBatchReports(10);

        // Link everything directly
        left->out.link(leftOut->input);
        right->out.link(rightOut->input);
        imu->out.link(imuOut->input);
    }

    void configureCam(std::shared_ptr<dai::node::MonoCamera> cam, dai::CameraBoardSocket socket, int fps) {
        cam->setBoardSocket(socket);
        cam->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
        cam->setFps(fps);
    }
};