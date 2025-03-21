#include "../User/SensorDriver.hpp"

#include "BST_SLAM/Solver.hpp"
#include "BST_SLAM/Messaging.hpp"

struct MapNode 
{
    cv::Mat Left;
    cv::Mat Right;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
};

int main() 
{    
    double RelMag = 0.025;
    double RelMax = 0.1;
    double MapMax = 500;

    SensorDriver* Sensor = nullptr;
    std::string ConnectFailed = "Failed to connect to sensor... e.what(): ";
    try { Sensor = new SensorDriver(); }
    catch (const std::exception& e) { std::cout << ConnectFailed << e.what() << std::endl; while (true) { ; } }
    cv::Vec3d CamPos;
    cv::Vec4d CamRot;

    BST_SLAM::Solver* Solver = new BST_SLAM::Solver();
    std::vector<MapNode> MapNodes;
    cv::Vec4d qvec;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    int MapIdx = -1;

    while (true)
    {
        SensorDriver::InputData InputData = SensorDriver::InputData();
        try { InputData = Sensor->GetInputData(); }
        catch (const std::exception& e) { std::cout << ConnectFailed << e.what() << std::endl; while (true) { ; } }
        if (InputData.Left.empty() || InputData.Right.empty() || InputData.Rot.empty() ||
            InputData.Rot.rows != 3 || InputData.Rot.cols != 3 ||
            InputData.Left.rows != InputData.Right.rows ||
            InputData.Left.cols != InputData.Right.cols) continue;
#ifndef NDEBUG
        std::cout << ConnectFailed << "Feature only available in release mode" << std::endl; while (true) { ; }
#endif
        InputData.Rot.convertTo(InputData.Rot, CV_64F);
        cv::Size sz = InputData.Left.size();

        static cv::Mat PrevLeft = InputData.Left.clone();
        static cv::Mat PrevRight = InputData.Right.clone();
        static cv::Mat PrevRot = InputData.Rot.clone();

        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0, 0) = InputData.fx;
        K.at<double>(1, 1) = InputData.fy;
        K.at<double>(0, 2) = sz.width / 2.0;
        K.at<double>(1, 2) = sz.height / 2.0;

        cv::Vec3d rvec1;
        cv::Rodrigues(PrevRot, rvec1);
        cv::Vec3d rvec2;
        cv::Rodrigues(InputData.Rot, rvec2);
        qvec = cv::Quatd::createFromRotMat(InputData.Rot.inv()).toVec();
        qvec = cv::Vec4d(qvec[1], qvec[2], qvec[3], qvec[0]);
        rvec = rvec2;
        bool WasSuccessful = false;
        cv::Vec3d TRel = Solver->SolveRelative
        (
            PrevLeft, PrevRight, rvec1,
            InputData.Left, InputData.Right, rvec2,
            K, InputData.Baseline, RelMax
        );
        if (cv::norm(TRel) > 0) tvec += TRel;

        if (MapNodes.empty() || cv::norm(MapNodes.back().tvec - tvec) >= RelMag) 
        {
            MapNode Node;
            Node.Left = InputData.Left.clone();
            Node.Right = InputData.Right.clone();
            Node.tvec = tvec;
            Node.rvec = rvec;
            MapNodes.push_back(Node);
            if (MapNodes.size() >= MapMax) MapNodes.erase(MapNodes.begin());
        }

        if (!MapNodes.empty())
        {
            if (MapIdx < MapNodes.size() - 1) MapIdx++;
            else MapIdx = 0;
            MapNode& Node = MapNodes[MapIdx];
            WasSuccessful = false;
            cv::Vec3d Offset = Solver->SolveRelative
            (
                Node.Left, Node.Right, Node.rvec,
                InputData.Left, InputData.Right, rvec,
                K, InputData.Baseline, RelMax
            );
            if (cv::norm(Offset) > 0)
            {
                cv::Vec3d Src = tvec;
                cv::Vec3d Dst = Node.tvec + Offset;
                tvec += (cv::normalize(Dst - Src) * 1e-6) * ((30.0 * MapMax) / 100.0);
            }
        }

        CamPos = cv::Vec3d(-tvec[0], tvec[1], -tvec[2]);
        CamRot = cv::Vec4d(-qvec[0], qvec[1], -qvec[2], -qvec[3]);

        BST_SLAM::SendMessage(CamPos, CamRot);

        InputData.Left.copyTo(PrevLeft);
        InputData.Right.copyTo(PrevRight);
        InputData.Rot.copyTo(PrevRot);
    }

    return 0;
}