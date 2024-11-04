#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>
#include <Sensor.hpp>
#include <Houdini.hpp>

namespace VIO {
	class Program {
	public:
		Sensor* Sensor = Sensor::GetInstance();
		static const bool ShouldTestSensor = false;

		cv::Mat Image;
		cv::Quatf RawRotation;

		cv::Vec3f Position;
		cv::Vec4f Rotation;

		int Main() {
			while (true) {
				Image = Sensor->GetImage(RawRotation);

				cv::Vec3f Translation = cv::detail::Houdini::tvec(Image, (cv::Mat)RawRotation.toRotMat3x3());
				Position += Translation;
				Rotation = GetRotation((cv::Mat)RawRotation.toRotMat3x3());

				std::cout << Position << Rotation << std::endl;
			}

			return 0;
		}

		static cv::Vec4f GetRotation(const cv::Mat& Rotation) {
			auto quaternion = cv::Quatf::createFromRotMat(Rotation);
			return cv::Vec4f(-quaternion.x, +quaternion.y, -quaternion.z, +quaternion.w);
		}
	};
}
