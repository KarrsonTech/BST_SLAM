#include "../User/SensorDriver.hpp"

#include "BST_SLAM/Solver.hpp"
#include "BST_SLAM/Messaging.hpp"
#include <opencv2/video/tracking.hpp>

struct MapNode 
{
    cv::Mat Left;
    cv::Mat Right;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
};

int main() 
{    
    double RelMax = 0.1;
    double RelMix = 0.9;
    double MapMax = 15;
    double MapMix1 = 0.5;
    double MapMix2 = 0.02;

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
    cv::Vec3d tvec1 = cv::Vec3d(0, 0, 0); 
    int MapIdx = -1;
    int stateSize = 6;    
    int measSize = 3;     
    int contrSize = 3;    
    cv::KalmanFilter KF(stateSize, measSize, contrSize, CV_64F);
    cv::Mat state(stateSize, 1, CV_64F);  
    cv::Mat meas(measSize, 1, CV_64F);    
    cv::Mat control(contrSize, 1, CV_64F); 
    cv::setIdentity(KF.transitionMatrix);
    KF.measurementMatrix = cv::Mat::zeros(measSize, stateSize, CV_64F);
    KF.measurementMatrix.at<double>(0,0) = 1.0;
    KF.measurementMatrix.at<double>(1,1) = 1.0;
    KF.measurementMatrix.at<double>(2,2) = 1.0;
    KF.controlMatrix = cv::Mat::zeros(stateSize, contrSize, CV_64F);
    KF.controlMatrix.at<double>(3,0) = 1.0;
    KF.controlMatrix.at<double>(4,1) = 1.0;
    KF.controlMatrix.at<double>(5,2) = 1.0;
    cv::setIdentity(KF.processNoiseCov, cv::Scalar((1.0 - MapMix1) / 100000000.0));
    KF.processNoiseCov.at<double>(3,3) = (1.0 - MapMix1) / 10000000.0; 
    KF.processNoiseCov.at<double>(4,4) = (1.0 - MapMix1) / 10000000.0; 
    KF.processNoiseCov.at<double>(5,5) = (1.0 - MapMix1) / 10000000.0; 
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar((1.0 - MapMix1) / 10000000.0));
    cv::setIdentity(KF.errorCovPost, cv::Scalar((1.0 - MapMix1) / 100000.0));
    double ticks = (double)cv::getTickCount();
    while (true)
    {
        double prevTick = ticks;
        ticks = (double)cv::getTickCount();
        double dt = (ticks - prevTick) / cv::getTickFrequency();
        double FpsMax = 120.0;
        double FpsMax2 = FpsMax * 2.0;
        while (dt <= 1.0 / FpsMax)
        {
            ticks = (double)cv::getTickCount();
            dt = (ticks - prevTick) / cv::getTickFrequency();
        }
        double Fps = 1.0 / dt;
        KF.transitionMatrix.at<double>(0,3) = Fps;
        KF.transitionMatrix.at<double>(1,4) = Fps;
        KF.transitionMatrix.at<double>(2,5) = Fps;
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
        cv::Vec3d TRel = dt > 0 ? Solver->SolveRelative(
            PrevLeft, PrevRight, rvec1,
            InputData.Left, InputData.Right, rvec2,
            K, InputData.Baseline, RelMax, RelMix
        ) / dt : cv::Vec3d();
        control.at<double>(0) = TRel[0];
        control.at<double>(1) = TRel[1];
        control.at<double>(2) = TRel[2];
        state = KF.predict(control);
        static cv::Vec3d measurement = tvec1;
        measurement += TRel * dt;
        static cv::Vec3d tvec;
        if (MapNodes.empty() || cv::norm(MapNodes.back().tvec - measurement) >= RelMax / 2.5) 
        {
            MapNode Node;
            Node.Left = InputData.Left.clone();
            Node.Right = InputData.Right.clone();
            Node.tvec = measurement;
            Node.rvec = rvec;
            bool AddIt = true;
            for (int i = MapNodes.size() - 1; i >= 0; i--)
            {
                const auto& MapNode = MapNodes[i];
                if (cv::norm(MapNode.tvec - measurement) <= RelMax / 2.5) {
                    AddIt = false;
                    break;
                }
            }
            if (AddIt)
            {
                MapNodes.push_back(Node);
                if (MapNodes.size() >= MapMax) MapNodes.erase(MapNodes.begin());
            }
        }
        if (!MapNodes.empty())
        {
            if (MapIdx < MapNodes.size() - 1) MapIdx++;
            else MapIdx = 0;
            MapNode& Node = MapNodes[MapIdx];
            cv::Vec3d Offset = Solver->SolveRelative(
                Node.Left, Node.Right, Node.rvec,
                InputData.Left, InputData.Right, rvec,
                K, InputData.Baseline, RelMax, RelMix
            );
            if (cv::norm(Offset) > 0) measurement += ((Node.tvec + Offset) - measurement) * MapMix2;
        }
        meas.at<double>(0) = measurement[0];
        meas.at<double>(1) = measurement[1];
        meas.at<double>(2) = measurement[2];
        state = KF.correct(meas);
        tvec1[0] = state.at<double>(0);
        tvec1[1] = state.at<double>(1);
        tvec1[2] = state.at<double>(2);
        static cv::Vec3d tvec2;
        tvec2 += TRel * dt;
        tvec2 += (tvec1 - tvec2) * (MapMix1 * (1.0 - Fps / FpsMax2));
        tvec += (tvec2 - tvec) * (RelMix * (1.0 - Fps / FpsMax2));

        CamPos = cv::Vec3d(-tvec[0], tvec[1], -tvec[2]);
        CamRot = cv::Vec4d(-qvec[0], qvec[1], -qvec[2], -qvec[3]);

        BST_SLAM::SendMessage(CamPos, CamRot);

        InputData.Left.copyTo(PrevLeft);
        InputData.Right.copyTo(PrevRight);
        InputData.Rot.copyTo(PrevRot);
    }

    return 0;
}