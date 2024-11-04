#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>
#include <Sensor.hpp>

namespace VIO {
    class SensorTest {
    public:
        Sensor* Sensor = Sensor::GetInstance();
        cv::Mat Image;
        cv::Quatf Rotation;

        int Main() {
            while (true) {
                Image = Sensor->GetImage(Rotation);

                cv::Mat DebugImage;
                cv::Mat Camera = cv::Mat::eye(3, 3, CV_32F);
                Camera.at<float>(0, 0) = Camera.at<float>(1, 1) = Image.size().width;
                Camera.at<float>(0, 2) = Camera.at<float>(1, 2) = Image.size().width / 2.0;

                cv::resize(Image, Image, cv::Size(Image.size().width, Image.size().width));
                cv::warpPerspective(Image, DebugImage, Camera * Rotation.toRotMat3x3() * Camera.inv(), Image.size());

                cv::imshow("The rotation should stabilize to zero. Please check.", DebugImage);
                cv::waitKey(1);
            }

            return 0;
        }
    };
}