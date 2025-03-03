#include "../User/SensorDriver.hpp"
#include "BST_SLAM/Solver.hpp"
#include "BST_SLAM/Messaging.hpp"
#include <opencv2/ximgproc.hpp>

float RelTPoseMax = 0.1f;
float NodeDist = 0.02f;
float MapMix = 0.02f;
int MapMax = 500;

SensorDriver* Sensor = new SensorDriver();

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

        static cv::Mat PrevLeft = InputData.Left.clone();
        static cv::Mat PrevRight = InputData.Right.clone();
        static cv::Mat PrevRot = InputData.Rot.clone();

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = InputData.fx;
        K.at<float>(1, 1) = InputData.fy;
        K.at<float>(0, 2) = sz.width / 2.0f;
        K.at<float>(1, 2) = sz.height / 2.0f;

        cv::Vec3f RelTPose = Solver->SolveISO3(PrevLeft, PrevRight, PrevRot, InputData.Left, InputData.Right, InputData.Rot, K, InputData.Baseline, RelTPoseMax);
        sz = InputData.Left.size();
        CamPos += RelTPose;

        if (MapNodes.empty() || cv::norm(MapNodes.back().Pos - CamPos) >= NodeDist) 
        {
            MapNode Node;
            Node.Left = InputData.Left.clone();
            Node.Right = InputData.Right.clone();
            Node.Pos = CamPos;
            Node.Rot = InputData.Rot.clone();
            MapNodes.push_back(Node);
            if (MapNodes.size() >= MapMax) MapNodes.erase(MapNodes.begin());
        }

        if (!MapNodes.empty())
        {
            MapIdx = (MapIdx + 1) % MapNodes.size();
            MapNode& Node = MapNodes[MapIdx];
            cv::Vec3f Offset = Solver->SolveISO3(Node.Left, Node.Right, Node.Rot, InputData.Left, InputData.Right, InputData.Rot, K, InputData.Baseline, RelTPoseMax);
            if (cv::norm(Offset) > 0) CamPos += (Node.Pos + Offset - CamPos) * MapMix;
        }

        auto Q = cv::Quatf::createFromRotMat(InputData.Rot.inv());
        cv::Vec4f CamRot(-Q.x, Q.y, -Q.z, -Q.w);

        BST_SLAM::SendMessage(CamPos, CamRot);

        cv::Mat DisparityMap;
        {
            cv::Mat LeftImgIn = InputData.Left.clone();
            cv::Mat RightImgIn = InputData.Right.clone();
            {
                cv::Mat LeftImgCopy = LeftImgIn.clone();
                cv::Mat RightImgCopy = RightImgIn.clone();
                cv::Ptr<cv::ximgproc::AdaptiveManifoldFilter> AdaptiveManifoldFilter = cv::ximgproc::createAMFilter(3, 0.7);
                cv::pyrDown(LeftImgIn, LeftImgIn);
                cv::pyrDown(RightImgIn, RightImgIn);
                AdaptiveManifoldFilter->filter(LeftImgIn, LeftImgIn);
                AdaptiveManifoldFilter->filter(RightImgIn, RightImgIn);
                cv::pyrUp(LeftImgIn, LeftImgIn);
                cv::pyrUp(RightImgIn, RightImgIn);
                cv::Mat LeftDisparityMap;
                cv::Mat RightDisparityMap;
                cv::Size TempSz = LeftImgIn.size();
                cv::resize(LeftImgIn, LeftImgIn, cv::Size(256, 256), 0, 0, cv::INTER_AREA);
                cv::resize(RightImgIn, RightImgIn, cv::Size(256, 256), 0, 0, cv::INTER_AREA);
                cv::Ptr<cv::StereoSGBM> StereoSGBM = cv::StereoSGBM::create();
                StereoSGBM->setMode(cv::StereoSGBM::MODE_HH4);
                StereoSGBM->compute(LeftImgIn, RightImgIn, LeftDisparityMap);
                StereoSGBM->compute(RightImgIn, LeftImgIn, RightDisparityMap);
                cv::Ptr<cv::ximgproc::DisparityWLSFilter> DisparityWLSFilter = cv::ximgproc::createDisparityWLSFilter(StereoSGBM);
                DisparityWLSFilter->setLRCthresh(255);
                DisparityWLSFilter->setLambda(24480);
                DisparityWLSFilter->filter(LeftDisparityMap, LeftImgIn, DisparityMap, RightDisparityMap, cv::Rect(), RightImgIn);
                cv::resize(DisparityMap, DisparityMap, TempSz, 0, 0, cv::INTER_AREA);
                LeftImgIn = LeftImgCopy.clone();
                RightImgIn = RightImgCopy.clone();
                int T = DisparityMap.type();
                DisparityMap.convertTo(DisparityMap, CV_8U);
                DisparityMap = 255 - DisparityMap;
                DisparityMap.convertTo(DisparityMap, CV_64F);
                DisparityMap.convertTo(DisparityMap, T);
                DisparityMap.convertTo(DisparityMap, CV_64F);
                DisparityMap = (DisparityMap / 255.0) * (double)LeftImgIn.size().width;
                DisparityMap.convertTo(DisparityMap, T);
            }
        }

        cv::Mat Vis = DisparityMap.clone();
        Vis.convertTo(Vis, CV_64F);
        Vis = (Vis / (double)InputData.Left.size().width) * 255.0;
        Vis.convertTo(Vis, CV_8U);
        cv::imshow("Vis", Vis);
        cv::waitKey(1);

        InputData.Left.copyTo(PrevLeft);
        InputData.Right.copyTo(PrevRight);
        InputData.Rot.copyTo(PrevRot);
    }

    return 0;
}