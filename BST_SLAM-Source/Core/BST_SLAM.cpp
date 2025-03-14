#include "../User/SensorDriver.hpp"

#include "BST_SLAM/Solver.hpp"
#include "BST_SLAM/Messaging.hpp"

struct MapNode 
{
    cv::Mat Left;
    cv::Mat Right;
    cv::Vec3f rvec;
    cv::Vec3f tvec;
};

int main() 
{
    float MinRel = 1e-7;
    float MaxRel = 0.10;
    float PtDist = 0.03;
    float MapMix = 0.03;
    float MapMax = 400;

    SensorDriver* Sensor = nullptr;
    std::string ConnectFailed = "Failed to connect to sensor... e.what(): ";
    try { Sensor = new SensorDriver(); }
    catch (const std::exception& e) { std::cout << ConnectFailed << e.what() << std::endl; while (true) { ; } }
    cv::Vec3f CamPos;
    cv::Vec4f CamRot;

    BST_SLAM::Solver* Solver = new BST_SLAM::Solver();
    std::vector<MapNode> MapNodes;
    cv::Vec4f qvec;
    cv::Vec3f rvec;
    cv::Vec3f tvec;
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
        InputData.Rot.convertTo(InputData.Rot, CV_32F);
        cv::Size sz = InputData.Left.size();

        static cv::Mat PrevLeft = InputData.Left.clone();
        static cv::Mat PrevRight = InputData.Right.clone();
        static cv::Mat PrevRot = InputData.Rot.clone();

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = InputData.fx;
        K.at<float>(1, 1) = InputData.fy;
        K.at<float>(0, 2) = sz.width / 2.0f;
        K.at<float>(1, 2) = sz.height / 2.0f;

        cv::Vec3f rvec1;
        cv::Rodrigues(PrevRot, rvec1);
        cv::Vec3f rvec2;
        cv::Rodrigues(InputData.Rot, rvec2);
        qvec = cv::Quatf::createFromRotMat(InputData.Rot.inv()).toVec();
        qvec = cv::Vec4f(qvec[1], qvec[2], qvec[3], qvec[0]);
        rvec = rvec2;
        tvec += Solver->SolveRelative
        (
            PrevLeft, PrevRight, rvec1, 
            InputData.Left, InputData.Right, rvec2, 
            K, InputData.Baseline, MinRel, MaxRel
        );

        if (MapNodes.empty() || cv::norm(MapNodes.back().tvec - tvec) >= PtDist) 
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
            cv::Vec3f Offset = Solver->SolveRelative
            (
                Node.Left, Node.Right, Node.rvec, 
                InputData.Left, InputData.Right, rvec, 
                K, InputData.Baseline, MinRel, MaxRel
            );
            if (cv::norm(Offset) > 0) tvec += (Node.tvec + Offset - tvec) * MapMix;
        }

        CamPos = cv::Vec3f(-tvec[0], tvec[1], -tvec[2]);
        CamRot = cv::Vec4f(-qvec[0], qvec[1], -qvec[2], -qvec[3]);

        BST_SLAM::SendMessage(CamPos, CamRot);

        InputData.Left.copyTo(PrevLeft);
        InputData.Right.copyTo(PrevRight);
        InputData.Rot.copyTo(PrevRot);
    }

    return 0;
}