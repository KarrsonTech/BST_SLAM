#pragma once
#include <opencv2/opencv.hpp>

namespace BST_SLAM {
    class Solver {
    public:
        cv::Vec3d SolveRelative( cv::Mat LeftImg1, cv::Mat RightImg1, cv::Vec3d rvec1,
                                 cv::Mat LeftImg2, cv::Mat RightImg2, cv::Vec3d rvec2,
                                 cv::Mat K, double Baseline, double Max ) {
            cv::Mat GlbRPose1;
            cv::Rodrigues(rvec1, GlbRPose1);
            cv::Mat GlbRPose2;
            cv::Rodrigues(rvec2, GlbRPose2);
            LeftImg1 = LeftImg1.clone();
            RightImg1 = RightImg1.clone();
            LeftImg2 = LeftImg2.clone();
            RightImg2 = RightImg2.clone();
            K = K.clone();
            PrepareImages(LeftImg1, RightImg1, LeftImg2, RightImg2, K, GlbRPose1, GlbRPose2);
            std::vector<cv::Point2d> A, B;
            if (!ExtractKeyPoints(LeftImg1, LeftImg2, A, B)) return cv::Vec3d();

            cv::Mat Mask;
            cv::Mat H = cv::findHomography(A, B, Mask, cv::USAC_MAGSAC);
            if (H.empty()) return cv::Vec3d();
            RemoveOutliers(A, B, Mask);
            if (A.size() < 15) return cv::Vec3d();
            H = cv::findHomography(A, B);
            if (H.empty()) return cv::Vec3d();
            H.convertTo(H, CV_64F);
            if (H.empty() || H.rows != 3 || H.cols != 3) return cv::Vec3d();

            cv::Mat Pts3D;
            if (!ComputePointCloud(LeftImg1, RightImg1, K, Baseline, Pts3D)) return cv::Vec3d();
            Pts3D.convertTo(Pts3D, CV_64F);
            cv::Vec3d RelTPose = SolveRelative(Pts3D, K, H);
            if (cv::norm(RelTPose) > Max) return cv::Vec3d();
            RelTPose = (cv::Mat)(GlbRPose1 * -RelTPose);
            return RelTPose;
        }

    private:
        double Resolution = 250;
        cv::Ptr<cv::ORB> ORB = cv::ORB::create(500, 1.65);
        cv::Ptr<cv::BFMatcher> BFMatcher = cv::BFMatcher::create(cv::NORM_HAMMING);

        void ResizeAndAdjustK( cv::Mat& Img, cv::Mat& K, cv::Size& Size,  bool SetK )
        {
            cv::resize(Img, Img, cv::Size(Resolution, Resolution), 0, 0, cv::INTER_AREA);

            if (SetK)
            {
                cv::Mat NewK = cv::Mat::eye(3, 3, CV_64F);
                NewK.at<double>(0, 0) = K.at<double>(0, 0) / Size.width * Resolution;
                NewK.at<double>(1, 1) = K.at<double>(1, 1) / Size.height * Resolution;
                NewK.at<double>(0, 2) = Resolution / 2.0f;
                NewK.at<double>(1, 2) = Resolution / 2.0f;

                K.release();
                K = NewK;

                Size = cv::Size(Resolution, Resolution);
            }
        }

        void PrepareImages( cv::Mat& LeftImg1, cv::Mat& RightImg1,
                            cv::Mat& LeftImg2, cv::Mat& RightImg2,
                            cv::Mat& K, cv::Mat& GlbRPose1, cv::Mat& GlbRPose2 )
        {
            cv::Size Size = LeftImg1.size();

            ResizeAndAdjustK(LeftImg1, K, Size, false);
            ResizeAndAdjustK(RightImg1, K, Size, false);
            ResizeAndAdjustK(LeftImg2, K, Size, false);
            ResizeAndAdjustK(RightImg2, K, Size, true);

            cv::Mat RelRPoseK = K * ((GlbRPose2.inv() * GlbRPose1).inv()).inv() * K.inv();

            cv::warpPerspective(LeftImg1, LeftImg1, RelRPoseK, LeftImg1.size());
            cv::warpPerspective(RightImg1, RightImg1, RelRPoseK, RightImg1.size());
        }

        bool ComputePointCloud( cv::Mat& LeftImg,  cv::Mat& RightImg,
                                cv::Mat& K, double Baseline, cv::Mat& PC3D )
        {
            std::vector<cv::KeyPoint> KpsL, KpsR;
            cv::Mat DesL, DesR;

            ORB->detectAndCompute(LeftImg, cv::noArray(), KpsL, DesL);
            ORB->detectAndCompute(RightImg, cv::noArray(), KpsR, DesR);

            if (KpsL.size() < 15 || KpsR.size() < 15) return false;

            std::vector<cv::DMatch> Matches;
            BFMatcher->match(DesL, DesR, Matches);
            if (Matches.size() < 15) return false;

            std::vector<cv::Point2d> PtsL, PtsR;

            for ( auto& M : Matches)
            {
                PtsL.push_back(KpsL[M.queryIdx].pt);
                PtsR.push_back(KpsR[M.trainIdx].pt);
            }

            cv::Mat InlierMask;
            if (cv::findHomography(PtsL, PtsR, InlierMask, cv::USAC_MAGSAC).empty()) return false;

            RemoveOutliers(PtsL, PtsR, InlierMask);

            if (PtsL.size() < 15) return false;

            cv::Mat ProjL = cv::Mat::eye(3, 4, CV_64F);
            ProjL.at<double>(0, 3) = Baseline / 2.0f;
            ProjL = K * ProjL;

            cv::Mat ProjR = cv::Mat::eye(3, 4, CV_64F);
            ProjR.at<double>(0, 3) = Baseline / -2.0f;
            ProjR = K * ProjR;

            cv::triangulatePoints(ProjL, ProjR, PtsL, PtsR, PC3D);

            if (PC3D.total() < 15) return false;

            cv::convertPointsFromHomogeneous(PC3D.t(), PC3D);
            PC3D.convertTo(PC3D, CV_64F);

            return PC3D.total() >= 15;
        }

        cv::Vec3d SolveRelative( cv::Mat& PC3D,  cv::Mat& K,  cv::Mat& RelTPoseK )
        {
            std::vector<cv::Point2d> Pts2DEye, Pts2DDiff;
            std::vector<cv::Point3d> PC3DEye(PC3D), PC3DDiff(PC3D);

            for (auto& P : PC3DEye) P.z *= 2.0f;
            for (auto& P : PC3DDiff) P.z *= 2.0f;

            cv::projectPoints(PC3DEye, cv::Vec3d(), cv::Vec3d(), K, cv::noArray(), Pts2DEye);

            if (Pts2DEye.size() < 15) return cv::Vec3d();

            cv::Vec3d RelTPoseEye, _;
            cv::solvePnP(PC3D, Pts2DEye, K, cv::noArray(), _, RelTPoseEye, true, cv::SOLVEPNP_ITERATIVE);

            cv::projectPoints(PC3DDiff, cv::Vec3d(), cv::Vec3d(), K, cv::noArray(), Pts2DDiff);

            if (!RelTPoseK.empty()) cv::perspectiveTransform(Pts2DDiff, Pts2DDiff, RelTPoseK);

            if (Pts2DDiff.size() < 15) return cv::Vec3d();

            cv::Vec3d RelTPoseDiff;
            cv::solvePnP(PC3D, Pts2DDiff, K, cv::Mat(), _, RelTPoseDiff, true, cv::SOLVEPNP_ITERATIVE);

            cv::Vec3d tvec = RelTPoseEye - RelTPoseDiff;
            return tvec;
        }

        bool ExtractKeyPoints( cv::Mat& Img1,  cv::Mat& Img2,
                               std::vector<cv::Point2d>& Pts1, 
                               std::vector<cv::Point2d>& Pts2 )
        {
            std::vector<cv::KeyPoint> Kps1, Kps2;
            cv::Mat Des1, Des2;

            ORB->detectAndCompute(Img1, cv::noArray(), Kps1, Des1);
            ORB->detectAndCompute(Img2, cv::noArray(), Kps2, Des2);

            if (Kps1.size() < 15 || Kps2.size() < 15) return false;

            std::vector<cv::DMatch> Matches;
            BFMatcher->match(Des1, Des2, Matches);

            if (Matches.size() < 15) return false;

            for ( auto& Match : Matches)
            {
                Pts1.push_back(Kps1[Match.queryIdx].pt);
                Pts2.push_back(Kps2[Match.trainIdx].pt);
            }

            return Pts1.size() >= 15 && Pts2.size() >= 15;
        }

        template <typename PtType> void RemoveOutliers
        ( std::vector<PtType>& Pts1, std::vector<PtType>& Pts2,  cv::Mat& InlierMask )
        {
            for (int i = InlierMask.total() - 1; i >= 0; i--)
            {
                if (!InlierMask.at<uchar>(i))
                {
                    Pts1.erase(Pts1.begin() + i);
                    Pts2.erase(Pts2.begin() + i);
                }
            }
        }
    };
}

