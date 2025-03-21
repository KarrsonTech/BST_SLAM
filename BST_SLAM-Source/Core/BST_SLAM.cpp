#include "../User/SensorDriver.hpp"
#include "BST_SLAM/Solver.hpp"
#include "BST_SLAM/Messaging.hpp"

float MaxTrans = 0.1f;
float NodeDist = 0.02f;
float MapMix = 0.02f;
int MapMax = 500;

SensorDriver* Sensor = new SensorDriver();
cv::Mat PrevLeft, PrevRight, PrevRot;

cv::Vec3f CamPos;
BST_SLAM::Solver* Solver = new BST_SLAM::Solver();

struct MapNode {
    cv::Mat Left;
    cv::Mat Right;
    cv::Vec3f Pos;
    cv::Mat Rot;
};

std::vector<MapNode> MapNodes;
int MapIdx = -1;

int main() {
    while (true) {
        auto InputData = Sensor->GetInputData();
        InputData.Rot.convertTo(InputData.Rot, CV_32F);
        cv::Size sz = InputData.Left.size();

        if (PrevRot.empty()) {
            PrevLeft = InputData.Left.clone();
            PrevRight = InputData.Right.clone();
            PrevRot = InputData.Rot.clone();
        }

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = InputData.fx;
        K.at<float>(1, 1) = InputData.fy;
        K.at<float>(0, 2) = sz.width / 2.0f;
        K.at<float>(1, 2) = sz.height / 2.0f;

        cv::Vec3f Trans = Solver->SolveISO3(PrevLeft, PrevRight, PrevRot, InputData.Left, InputData.Right, InputData.Rot, K, InputData.Baseline, MaxTrans);
        CamPos += Trans;

        if (MapNodes.empty() || cv::norm(MapNodes.back().Pos - CamPos) >= NodeDist) {
            MapNode Node;
            Node.Left = InputData.Left.clone();
            Node.Right = InputData.Right.clone();
            Node.Pos = CamPos;
            Node.Rot = InputData.Rot.clone();
            MapNodes.push_back(Node);
            if (MapNodes.size() > MapMax) MapNodes.erase(MapNodes.begin());
        }

        MapIdx = (MapIdx + 1) % MapNodes.size();
        MapNode& Node = MapNodes[MapIdx];
        cv::Vec3f Offset = Solver->SolveISO3(Node.Left, Node.Right, Node.Rot, InputData.Left, InputData.Right, InputData.Rot, K, InputData.Baseline, MaxTrans);
        if (cv::norm(Offset) > 0) CamPos += (Node.Pos + Offset - CamPos) * MapMix;

        PrevLeft = InputData.Left.clone();
        PrevRight = InputData.Right.clone();
        PrevRot = InputData.Rot.clone();

        auto Q = cv::Quatf::createFromRotMat(InputData.Rot.inv());
        cv::Vec4f CamRot(-Q.x, Q.y, -Q.z, -Q.w);

        BST_SLAM::SendMessage(CamPos, CamRot);
    }

    return 0;
}