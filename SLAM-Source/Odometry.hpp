#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>
#include <memory>
#include <optional>

using namespace std;
using namespace cv;

class Odometry {
public:
    void DebugInputs(Mat CurrImage, Mat CurrRotation) const {
        Mat DebugImage;

        CurrImage = CurrImage.clone();
        CurrRotation = CurrRotation.clone();

        Mat Camera = Mat::eye(3, 3, CV_64F);
        Camera.at<double>(0, 0) = Camera.at<double>(1, 1) = Resolution;
        Camera.at<double>(0, 2) = Camera.at<double>(1, 2) = Resolution / 2.0;

        Mat CurrRotation64 = CurrRotation.clone();
        CurrRotation64.convertTo(CurrRotation64, CV_64F);

        resize(CurrImage, CurrImage, Size(Resolution, Resolution));
        warpPerspective(CurrImage, DebugImage,
            Camera * CurrRotation64 * Camera.inv(),
            CurrImage.size());

        imshow("The rotation should stabilize to zero. Please check.", DebugImage);
    }

    Vec3f EstimateTranslation(Mat CurrLeft, Mat CurrRight, float Baseline, Mat CurrRotation) {
        if (CurrLeft.empty() || CurrRight.empty() || Baseline <= FLT_EPSILON || CurrRotation.empty()) return Vec3f();

        CurrLeft = CurrLeft.clone();
        CurrRight = CurrRight.clone();
        CurrRotation = CurrRotation.clone();
        CurrRotation.convertTo(CurrRotation, CV_32F);
        T0 = Vec3f();

        try {
            if (PrevLeft.empty()) PrevLeft = CurrLeft.clone();
            if (PrevRight.empty()) PrevRight = CurrRight.clone();
            if (PrevRotation.empty()) PrevRotation = CurrRotation.clone();
            PrevRotation.convertTo(PrevRotation, CV_32F);
            CurrRotation.convertTo(CurrRotation, CV_32F);

            if (PrevLeft.channels() != 1) cvtColor(PrevLeft, PrevLeft, COLOR_BGR2GRAY);
            if (PrevRight.channels() != 1) cvtColor(PrevRight, PrevRight, COLOR_BGR2GRAY);
            if (CurrLeft.channels() != 1) cvtColor(CurrLeft, CurrLeft, COLOR_BGR2GRAY);
            if (CurrRight.channels() != 1) cvtColor(CurrRight, CurrRight, COLOR_BGR2GRAY);

            resize(PrevLeft, PrevLeft, Size(Resolution, Resolution), 0, 0, INTER_AREA);
            resize(PrevRight, PrevRight, Size(Resolution, Resolution), 0, 0, INTER_AREA);
            resize(CurrLeft, CurrLeft, Size(Resolution, Resolution), 0, 0, INTER_AREA);
            resize(CurrRight, CurrRight, Size(Resolution, Resolution), 0, 0, INTER_AREA);

            Mat RelRotation = CurrRotation * PrevRotation.inv();
            RelRotation.convertTo(RelRotation, CV_32F);

            Mat Camera = Mat::eye(3, 3, CV_32F);
            Camera.at<float>(0, 0) = Resolution;
            Camera.at<float>(1, 1) = Resolution;
            Camera.at<float>(0, 2) = Resolution / 2.0;
            Camera.at<float>(1, 2) = Resolution / 2.0;

            warpPerspective(PrevLeft, PrevLeft, Camera * RelRotation * Camera.inv(), PrevLeft.size());
            warpPerspective(PrevRight, PrevRight, Camera * RelRotation * Camera.inv(), PrevRight.size());

            Vec3f rvec;
            Boss1->track(CurrLeft.clone(), CurrRight.clone(), Baseline, CurrRotation.clone(), rvec, T0);
            T0 = -(Vec3f)(Mat)(CurrRotation * Vec3f(T0));
            T0[1] *= -1;
        }
        catch (...) { ; }
        T1 += (T0 - T1) * signal_to_noise2;
        Vec3f tvec = T1;

        PrevLeft = CurrLeft.clone();
        PrevRight = CurrRight.clone();
        PrevRotation = CurrRotation.clone();
        return tvec;
    }

    Vec4f QuaternionFromRotation(const Mat& rotation) {
        auto quaternion = Quatf::createFromRotMat(rotation);
        return Vec4f(-quaternion.x, +quaternion.y, -quaternion.z, +quaternion.w);
    }

private:
    static constexpr float Resolution = 320;
    static constexpr float MaxSpeed = 200;
    static constexpr float signal_to_noise1 = 0.99999f;
    static constexpr float signal_to_noise2 = 0.8f;

    Vec3f T0;
    Vec3f T1;

    class OdometryBossLevel1 {
    private:
        struct GlobalPose {
            cv::Vec3f curr_tvec, curr_rvec;
            cv::Vec3f prev_tvec, prev_rvec;
            cv::Vec3f saved_tvec, saved_rvec;
        };

        struct GlobalState {
            GlobalPose internal;
            GlobalPose high_level;
            cv::Mat motion_2d = cv::Mat::eye(3, 3, CV_32F);
        };

        cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_32F);
        float baseline = 0;
        cv::Vec3f rel_rvec, rel_tvec;
        std::vector<cv::Point3f> initial_points;
        GlobalState state;
        cv::Mat saved_imu_rotation_cloud = cv::Mat::eye(3, 3, CV_32F);
        cv::Mat curr_imu_rotation_cloud = cv::Mat::eye(3, 3, CV_32F);
        cv::Mat prev_imu_rotation_motion = cv::Mat::eye(3, 3, CV_32F);
        cv::Mat curr_imu_rotation_motion = cv::Mat::eye(3, 3, CV_32F);
        std::vector<cv::Point3f> saved_valid_points;
        cv::Mat prevLeft;
        bool is_init = true;
        cv::Vec3f tvec1, rvec1;
        cv::Vec3f tvec2, rvec2;

        std::vector<cv::Point3f> generatePointCloud(const cv::Mat& left, const cv::Mat& right) {
            cv::Ptr<cv::ORB> feature_detector = cv::ORB::create();
            cv::BFMatcher matcher;

            std::vector<cv::KeyPoint> kpts_left, kpts_right;
            cv::Mat desc_left, desc_right;
            feature_detector->detectAndCompute(left, cv::noArray(), kpts_left, desc_left);
            feature_detector->detectAndCompute(right, cv::noArray(), kpts_right, desc_right);

            if (desc_left.empty() || desc_right.empty()) throw new std::runtime_error("");

            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher.knnMatch(desc_left, desc_right, knn_matches, 2);

            std::vector<cv::Point2f> points_left, points_right;
            for (const auto& match : knn_matches) {
                if (match.size() >= 2 && match[0].distance <= match[1].distance * 0.9995f) {
                    points_left.push_back(kpts_left[match[0].queryIdx].pt);
                    points_right.push_back(kpts_right[match[0].trainIdx].pt);
                }
            }

            if (points_left.size() < 15) throw new std::runtime_error("");

            cv::Mat mask;
            cv::findHomography(points_left, points_right, mask, cv::USAC_MAGSAC);

            std::vector<cv::Point2f> filtered_left, filtered_right;
            for (size_t i = 0; i < mask.total(); i++) {
                if (mask.at<uchar>(i)) {
                    filtered_left.push_back(points_left[i]);
                    filtered_right.push_back(points_right[i]);
                }
            }

            if (filtered_left.size() < 15) throw new std::runtime_error("");

            cv::Mat P1 = cv::Mat::eye(3, 4, CV_32F);
            P1.at<float>(0, 3) = +baseline;
            P1 = camera_matrix * P1;

            cv::Mat P2 = cv::Mat::eye(3, 4, CV_32F);
            P2.at<float>(0, 3) = -baseline;
            P2 = camera_matrix * P2;

            cv::Mat points_4d;
            cv::triangulatePoints(P1, P2, filtered_left, filtered_right, points_4d);

            cv::Mat points_3d;
            cv::convertPointsFromHomogeneous(points_4d.t(), points_3d);

            std::vector<cv::Point3f> point_cloud;
            point_cloud.reserve(points_3d.rows);
            for (int i = 0; i < points_3d.rows; i++) point_cloud.push_back(points_3d.at<cv::Point3f>(i));

            return point_cloud;
        }

        cv::Mat get2DMotion(const cv::Mat& prev_frame, const cv::Mat& curr_frame) {
            cv::Ptr<cv::ORB> feature_detector = cv::ORB::create();
            cv::BFMatcher matcher;

            std::vector<cv::KeyPoint> kpts1, kpts2;
            cv::Mat desc1, desc2;
            feature_detector->detectAndCompute(prev_frame, cv::noArray(), kpts1, desc1);
            feature_detector->detectAndCompute(curr_frame, cv::noArray(), kpts2, desc2);

            if (desc1.empty() || desc2.empty()) throw new std::runtime_error("");

            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher.knnMatch(desc1, desc2, knn_matches, 2);

            std::vector<cv::Point2f> points1, points2;
            for (const auto& match : knn_matches) {
                if (match.size() >= 2 && match[0].distance <= match[1].distance * 0.95f) {
                    points1.push_back(kpts1[match[0].queryIdx].pt);
                    points2.push_back(kpts2[match[0].trainIdx].pt);
                }
            }

            if (points1.size() < 15) throw new std::runtime_error("");

            cv::Mat mask;
            cv::findHomography(points1, points2, mask, cv::USAC_MAGSAC);

            std::vector<cv::Point2f> filtered_points1, filtered_points2;
            for (size_t i = 0; i < mask.total(); i++) {
                if (mask.at<uchar>(i)) {
                    filtered_points1.push_back(points1[i]);
                    filtered_points2.push_back(points2[i]);
                }
            }

            if (filtered_points1.size() < 15) throw new std::runtime_error("");

            cv::Mat affine = cv::estimateAffinePartial2D(filtered_points1, filtered_points2);
            if (affine.empty()) throw new std::runtime_error("");

            cv::Mat H_translation;
            affine.convertTo(affine, CV_32F);
            cv::vconcat(affine, cv::Mat::zeros(1, 3, CV_32F), H_translation);
            H_translation.at<float>(2, 2) = 1.0f;

            return H_translation;
        }

        std::vector<cv::Point2f> projectPoints(const std::vector<cv::Point3f>& points, const cv::Mat& motion) {
            std::vector<cv::Point2f> projected_points;
            cv::projectPoints(points, cv::Vec3f(), cv::Vec3f(), camera_matrix, cv::noArray(), projected_points);
            if (!motion.empty()) cv::perspectiveTransform(projected_points, projected_points, motion);
            return projected_points;
        }

        void saveKeyFrame(const std::vector<cv::Point3f>& new_points) {
            initial_points = new_points;
            state.internal.saved_tvec = state.high_level.curr_tvec;
            state.internal.saved_rvec = state.high_level.curr_rvec;
            state.motion_2d = cv::Mat::eye(3, 3, CV_32F);
        }

    public:
        void init(const cv::Mat& left, const cv::Mat& right) {
            auto point_cloud = generatePointCloud(left, right);
            if (point_cloud.empty()) throw new runtime_error("");
            initial_points = point_cloud;
            state = GlobalState();
            state.motion_2d = cv::Mat::eye(3, 3, CV_32F);
        }

        clock_t t0;
        float DT = -1;

        void track(const cv::Mat& curr_left, const cv::Mat& curr_right,
            const float& curr_baseline, const cv::Mat& curr_rot,
            cv::Vec3f& out_rvec, cv::Vec3f& out_tvec) {

            rel_tvec = cv::Vec3f();
            rel_rvec = cv::Vec3f();

            if (is_init) {
                camera_matrix = cv::Mat::eye(3, 3, CV_32F);
                camera_matrix.at<float>(0, 0) = curr_left.cols;
                camera_matrix.at<float>(1, 1) = curr_left.rows;
                camera_matrix.at<float>(0, 2) = curr_left.cols / 2.0f;
                camera_matrix.at<float>(1, 2) = curr_left.rows / 2.0f;
                baseline = curr_baseline;
                saved_imu_rotation_cloud = curr_rot.clone();
                curr_imu_rotation_cloud = curr_rot.clone();
                prev_imu_rotation_motion = curr_rot.clone();
                curr_imu_rotation_motion = curr_rot.clone();
                prevLeft = curr_left.clone();
                init(curr_left, curr_right);
            }

            try {
                if (is_init) {
                    is_init = false;
                    throw new runtime_error("");
                }
                curr_imu_rotation_motion = curr_rot.clone();
                curr_imu_rotation_cloud = curr_rot.clone();

                auto motion_2d = get2DMotion(prevLeft, curr_left);
                state.motion_2d = motion_2d * state.motion_2d;

                Vec3f prev_tvec1 = tvec1;
                tvec1[2] *= 2;
                cv::solvePnP(initial_points, projectPoints(initial_points, state.motion_2d),
                    camera_matrix, cv::noArray(), rvec1, tvec1, false, SOLVEPNP_ITERATIVE);
                tvec1[2] *= 0.5;
                if (cv::norm(tvec1) <= FLT_EPSILON) tvec1 = prev_tvec1;
                Vec3f T = tvec1;
                tvec1 = tvec2 * (1.0 - std::fmin(cv::norm(tvec1) / MaxSpeed * DT, 1)) + tvec1 * std::fmin(cv::norm(tvec1) / MaxSpeed * DT, 1);
                if (!std::isnormal(tvec1[0]) || !std::isnormal(tvec1[1]) || !std::isnormal(tvec1[2])) tvec1 = tvec2;

                tvec1 += state.internal.saved_tvec;
                rvec1 += state.internal.saved_rvec;

                auto new_points = generatePointCloud(curr_left, curr_right);

                saveKeyFrame(new_points);

                state.internal.curr_tvec += tvec1 - state.internal.prev_tvec;
                state.internal.curr_rvec += rvec1 - state.internal.prev_rvec;

                state.internal.prev_tvec = tvec1;
                state.internal.prev_rvec = rvec1;

                Vec3f prev_tvec2 = tvec2;
                tvec2[2] *= 2;
                cv::solvePnP(new_points, projectPoints(new_points, motion_2d),
                    camera_matrix, cv::noArray(), rvec2, tvec2, true, SOLVEPNP_ITERATIVE);
                tvec2[2] *= 0.5;
                if (cv::norm(tvec2) <= FLT_EPSILON) tvec2 = prev_tvec2;
                if (DT < 0) t0 = clock();
                DT = clock() - t0;
                while (DT <= 0) DT = clock() - t0;
                t0 = clock();
                if (cv::norm(tvec2) >= MaxSpeed * DT) tvec2 = Vec3f();
                tvec2 = T * (1.0 - std::fmin(cv::norm(tvec2) / MaxSpeed * DT, 1)) + tvec2 * std::fmin(cv::norm(tvec2) / MaxSpeed * DT, 1);
                if (!std::isnormal(tvec2[0]) || !std::isnormal(tvec2[1]) || !std::isnormal(tvec2[2])) tvec2 = T;


                state.internal.curr_tvec =
                    (state.high_level.prev_tvec + tvec2) * signal_to_noise1 +
                    state.internal.curr_tvec * (1.0f - signal_to_noise1);

                state.internal.curr_rvec =
                    (state.high_level.prev_rvec + rvec2) * signal_to_noise1 +
                    state.internal.curr_rvec * (1.0f - signal_to_noise1);

                state.high_level.curr_tvec = state.internal.curr_tvec;
                state.high_level.curr_rvec = state.internal.curr_rvec;

                rel_rvec = state.high_level.curr_rvec - state.high_level.prev_rvec;
                rel_tvec = state.high_level.curr_tvec - state.high_level.prev_tvec;

                state.high_level.prev_tvec = state.high_level.curr_tvec;
                state.high_level.prev_rvec = state.high_level.curr_rvec;
            }
            catch (...) { ; }

            if (cv::norm(rel_rvec) <= FLT_EPSILON) {
                rvec1 = cv::Vec3f();
                rvec2 = cv::Vec3f();
            }

            if (cv::norm(rel_tvec) <= FLT_EPSILON) {
                tvec1 = cv::Vec3f();
                tvec2 = cv::Vec3f();
            }

            out_rvec = rel_rvec;
            out_tvec = rel_tvec;

            prev_imu_rotation_motion = curr_rot.clone();
            prevLeft = curr_left.clone();
        }
    };

    OdometryBossLevel1* Boss1 = new OdometryBossLevel1();

    Mat PrevLeft;
    Mat PrevRight;
    Mat PrevRotation;
};