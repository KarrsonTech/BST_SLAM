#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>
#include <depthai/depthai.hpp>

namespace VIO {
	class Sensor {
	private:
		dai::Device NativeDevice;
		dai::Pipeline NativePipeline;
		std::shared_ptr<dai::DataOutputQueue> NativeCamera_Q;
		std::shared_ptr<dai::DataOutputQueue> NativeIMU_Q;
		cv::Mat RotationInit;

		void Initialize() {
			auto NativeCamera = NativePipeline.create<dai::node::MonoCamera>();
			auto NativeCamera_X = NativePipeline.create<dai::node::XLinkOut>();
			NativeCamera_X->setStreamName("NativeCamera_X");
			NativeCamera->setBoardSocket(dai::CameraBoardSocket::CAM_B);
			NativeCamera->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
			NativeCamera->setFps(20);
			NativeCamera->out.link(NativeCamera_X->input);

			auto NativeIMU = NativePipeline.create<dai::node::IMU>();
			auto NativeIMU_X = NativePipeline.create<dai::node::XLinkOut>();
			NativeIMU_X->setStreamName("NativeIMU_X");
			NativeIMU->enableIMUSensor(dai::IMUSensor::ROTATION_VECTOR, 400);
			NativeIMU->setBatchReportThreshold(1);
			NativeIMU->setMaxBatchReports(10);
			NativeIMU->out.link(NativeIMU_X->input);

			NativeDevice.startPipeline(NativePipeline);
			NativeCamera_Q = NativeDevice.getOutputQueue("NativeCamera_X", 1, false);
			NativeIMU_Q = NativeDevice.getOutputQueue("NativeIMU_X", 1, false);
		}

	public:
		static Sensor* GetInstance() {
			static Sensor* SensorInstance = new Sensor();
			SensorInstance->Initialize();
			return SensorInstance;
		}

		cv::Mat GetImage(cv::Quatf& Rotation) {
			auto IMU = NativeIMU_Q->get<dai::IMUData>()->packets.back();
			auto RotationVector = IMU.rotationVector;
			cv::Mat _Rotation = (cv::Mat)cv::Quatf(+RotationVector.k, -RotationVector.i, -RotationVector.j, +RotationVector.real).toRotMat3x3();
			if (RotationInit.empty()) RotationInit = _Rotation.clone();
			Rotation = cv::Quatf::createFromRotMat((_Rotation * RotationInit.inv()).inv());
			return NativeCamera_Q->get<dai::ImgFrame>()->getCvFrame();
		}
	};
}