#include "OAK_D.hpp"
#include <BST_SLAM/BST_SLAM.hpp>

OAK_D* Sensor = new OAK_D();

int main()
{
    while (true) 
    {
        // Input
        auto SensorFrame = Sensor->ReadSensorFrame();

        // Process
        cv::Vec3f CurrentWorldSpaceSensorPosition = BST_SLAM::CalculateCurrentWorldSpaceSensorPosition
        (SensorFrame.Rotation, SensorFrame.fx, SensorFrame.fy, SensorFrame.Left, SensorFrame.Right, SensorFrame.StereoBaselineDistance);
        cv::Vec4f CurrentWorldSpaceSensorRotation = BST_SLAM::ConvertRotation(SensorFrame.Rotation);

        // Output
        std::cout << CurrentWorldSpaceSensorPosition << CurrentWorldSpaceSensorRotation << std::endl;
    }
}