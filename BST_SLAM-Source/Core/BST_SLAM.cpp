#include "../User/SensorDriver.hpp"

#include "BST_SLAM/Solver.hpp"
#include "BST_SLAM/Messaging.hpp"

void prediction_update(cv::Mat& mu, cv::Mat& sigma, cv::Mat& u, int& n_state, cv::Mat& Fx, cv::Mat& R) {
    double rx = mu.at<double>(0);
    double ry = mu.at<double>(1);
    double rz = mu.at<double>(2);
    
    double vw_x = u.at<double>(0);
    double vw_y = u.at<double>(1);
    double vw_z = u.at<double>(2);

    cv::Mat state_model_mat = cv::Mat::zeros(n_state, 1, CV_64F);
    state_model_mat.at<double>(0) = vw_x;
    state_model_mat.at<double>(1) = vw_y;
    state_model_mat.at<double>(2) = vw_z;
    mu += Fx.t() * state_model_mat;

    cv::Mat state_jacobian_mat = cv::Mat::zeros(n_state, n_state, CV_64F);
    state_jacobian_mat.at<double>(0, 2) = vw_x;
    state_jacobian_mat.at<double>(1, 2) = vw_y;
    state_jacobian_mat.at<double>(2, 2) = vw_z;

    cv::Mat G = cv::Mat::eye(sigma.rows, sigma.rows, CV_64F) + Fx.t() * state_jacobian_mat * Fx;
    sigma = G * sigma * G.t() + Fx.t() * R * Fx;
}

void sigma2transform(cv::Mat& sigma_sub, cv::Mat& eigenvals, double& angle, int XIdx=0, int YIdx=2) {
    cv::Mat eigenvecs;
    cv::eigen(sigma_sub, eigenvals, eigenvecs);
    eigenvecs.convertTo(eigenvecs, CV_64F);
    angle = 180.0 * std::atan2(eigenvecs.at<double>(YIdx, 0), eigenvecs.at<double>(XIdx, 0)) / CV_PI;
}

int main() 
{    
    double RelMax = 0.1;

    SensorDriver* Sensor = nullptr;
    std::string ConnectFailed = "Failed to connect to sensor... e.what(): ";
    try { Sensor = new SensorDriver(); }
    catch (const std::exception& e) { std::cout << ConnectFailed << e.what() << std::endl; while (true) { ; } }
    cv::Vec3d CamPos;
    cv::Vec4d CamRot;

    BST_SLAM::Solver* Solver = new BST_SLAM::Solver();
    cv::Vec4d qvec;
    cv::Vec3d rvec;
    cv::Vec3d tvec;

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
        static int n_state = 3;
        static int n_landmarks = 1;
        static cv::Mat mu = cv::Mat::zeros(n_state + n_state * n_landmarks, 1, CV_64F);
        static cv::Mat sigma = cv::Mat::zeros(n_state + n_state * n_landmarks, n_state + n_state * n_landmarks, CV_64F);
        static cv::Mat Fx;
        static double ErrLo = 0.005;
        static double ErrHi = 0.025;
        static cv::Mat R = cv::Mat::eye(n_state, n_state, CV_64F) * ErrLo;
        static bool init_ekf = true;
        if (init_ekf) {
            Fx = cv::Mat::eye(3, 3, CV_64F);
            cv::hconcat(Fx, cv::Mat::zeros(3, n_state * n_landmarks, CV_64F), Fx);
            sigma = cv::Mat::eye(sigma.rows, sigma.cols, CV_64F) * 300.0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    sigma.at<double>(i, j) = ErrHi;
                }
            }
        }
        cv::Mat u = cv::Mat::zeros(n_state, 1, CV_64F);
        u.at<double>(0) = TRel[0];
        u.at<double>(1) = TRel[1];
        u.at<double>(2) = TRel[2];
        prediction_update(mu, sigma, u, n_state, Fx, R);
        tvec = cv::Vec3d(mu.at<double>(0), mu.at<double>(1), mu.at<double>(2));

        CamPos = cv::Vec3d(-tvec[0], tvec[1], -tvec[2]);
        CamRot = cv::Vec4d(-qvec[0], qvec[1], -qvec[2], -qvec[3]);

        BST_SLAM::SendMessage(CamPos, CamRot);

        InputData.Left.copyTo(PrevLeft);
        InputData.Right.copyTo(PrevRight);
        InputData.Rot.copyTo(PrevRot);
    }

    return 0;
}