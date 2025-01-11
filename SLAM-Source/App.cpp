#include "OAK_D.hpp"
#include <Odometry.hpp>

OAK_D* Sensor = new OAK_D();

int main()
{
	Odometry VIO;

    while (true) 
    {
        // Input
        auto Frame = Sensor->ReadSensorFrame();

        // Process
        static cv::Vec3f Pos;
        Pos += VIO.EstimateTranslation(Frame.Left, Frame.Right, Frame.StereoBaselineDistance, Frame.Rotation);
        cv::Vec4f Rot = VIO.QuaternionFromRotation(Frame.Rotation);

        // Output
        std::cout << Pos << Rot << std::endl;
    }
}