#include "OAK_D.hpp"
#include <BST_SLAM/BST_SLAM.hpp>

OAK_D* Sensor = new OAK_D();

int main()
{
    while (true) 
    {
        // Input
        auto Frame = Sensor->ReadSensorFrame();

        // Process
        cv::Vec3f Pos = BST_SLAM::EstimatePosition3D(Frame.Left, Frame.Rotation);
        cv::Vec4f Rot = BST_SLAM::QuaternionFromRotation(Frame.Rotation);

        // Output
        std::cout << Pos << Rot << std::endl;
    }
}