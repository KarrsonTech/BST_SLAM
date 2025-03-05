#include "../User/SensorDriver.hpp"
#include "BST_SLAM/Solver.hpp"
#include "BST_SLAM/Messaging.hpp"
#include <opencv2/ximgproc.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <unordered_map>
#include <cmath>
#include <iostream>

float RelTPoseMin = 0.00015;
float RelTPoseMax = 0.1f;
float Smoother = 0.98;
float NodeDist = 0.02f;
float MapMix = 0.02f;
int MapMax = 500;
int MapMax3D = 25000;

SensorDriver* Sensor = new SensorDriver();

cv::Vec4f qvec;
cv::Vec3f rvec;
cv::Vec3f tvec;
BST_SLAM::Solver* Solver = new BST_SLAM::Solver();
cv::Vec3f CamPos;
cv::Vec4f CamRot;

struct MapNode {
    cv::Mat Left;
    cv::Mat Right;
    cv::Vec3f rvec;
    cv::Vec3f tvec;
};

std::vector<MapNode> MapNodes;
int MapIdx = -1;

int main() {
    while (true) {
        auto InputData = Sensor->GetInputData();
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
        tvec += Solver->SolveISO3(PrevLeft, PrevRight, rvec1, InputData.Left, InputData.Right, rvec2, K, InputData.Baseline, RelTPoseMin, RelTPoseMax);

        if (MapNodes.empty() || cv::norm(MapNodes.back().tvec - tvec) >= NodeDist) 
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
            MapIdx = (MapIdx + 1) % MapNodes.size();
            MapNode& Node = MapNodes[MapIdx];
            cv::Vec3f Offset = Solver->SolveISO3(Node.Left, Node.Right, Node.rvec, InputData.Left, InputData.Right, rvec, K, InputData.Baseline, RelTPoseMin, RelTPoseMax);
            if (cv::norm(Offset) > 0) tvec += (Node.tvec + Offset - tvec) * MapMix;
        }

        CamPos += (cv::Vec3f(-tvec[0], tvec[1], -tvec[2]) - CamPos) * Smoother;
        CamRot = cv::Vec4f(-qvec[0], qvec[1], -qvec[2], -qvec[3]);

        BST_SLAM::SendMessage(CamPos, CamRot);

        InputData.Left.copyTo(PrevLeft);
        InputData.Right.copyTo(PrevRight);
        InputData.Rot.copyTo(PrevRot);
    }

    return 0;
}