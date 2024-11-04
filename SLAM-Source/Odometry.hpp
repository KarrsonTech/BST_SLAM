#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>

using namespace std;
using namespace cv;

class Odometry {
private:
    const float Resolution = 360;

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

    Vec3f EstimateTranslation(Mat CurrImage, Mat CurrRotation) {
        CurrImage = CurrImage.clone();
        CurrRotation = CurrRotation.clone();
        CurrRotation.convertTo(CurrRotation, CV_32F);

        Vec3f RawTranslation;

        try {
            if (PrevImage.empty()) PrevImage = CurrImage;
            if (PrevRotation.empty()) PrevRotation = CurrRotation;
            PrevRotation.convertTo(PrevRotation, CV_32F);
            CurrRotation.convertTo(CurrRotation, CV_32F);

            resize(PrevImage, PrevImage, Size(Resolution, Resolution), 0, 0, INTER_AREA);
            resize(CurrImage, CurrImage, Size(Resolution, Resolution), 0, 0, INTER_AREA);

            Mat RelRotation = (PrevRotation.inv() * CurrRotation).inv();
            RelRotation.convertTo(RelRotation, CV_32F);

            Mat Camera = Mat::eye(3, 3, CV_32F);
            Camera.at<float>(0, 0) = Camera.at<float>(1, 1) = Resolution;
            Camera.at<float>(0, 2) = Camera.at<float>(1, 2) = Resolution / 2.0;

            warpPerspective(PrevImage, PrevImage, Camera * RelRotation * Camera.inv(), CurrImage.size());

            vector<KeyPoint> Keypoints1;
            Mat Descriptors1;
            FeatureDetector->detectAndCompute(PrevImage, noArray(), Keypoints1, Descriptors1);

            vector<KeyPoint> Keypoints2;
            Mat Descriptors2;
            FeatureDetector->detectAndCompute(CurrImage, noArray(), Keypoints2, Descriptors2);

            vector<DMatch> Matches;
            FindMatches(Matches, Keypoints1, Descriptors1, Keypoints2, Descriptors2);

            vector<Point2f> Points1, Points2;
            for (DMatch& Match : Matches) {
                Points1.push_back(Keypoints1[Match.queryIdx].pt);
                Points2.push_back(Keypoints2[Match.trainIdx].pt);
            }

            Mat Affine = estimateAffinePartial2D(Points1, Points2);
            Affine.convertTo(Affine, CV_32F);
            if (Affine.empty()) Affine = Mat::eye(3, 3, CV_32F);
            else vconcat(Affine, Vec3f(0, 0, 1).t(), Affine);

            vector<Point3f> Src = {
                Point3f(-ArbitraryParameter5, +ArbitraryParameter5, 0), Point3f(+ArbitraryParameter5, +ArbitraryParameter5, 0),
                Point3f(-ArbitraryParameter5, -ArbitraryParameter5, 0), Point3f(+ArbitraryParameter5, -ArbitraryParameter5, 0)
            };
            vector<Point2f> Dst;
            projectPoints(Src, cv::Vec3f(), cv::Vec3f(), Camera, cv::Mat(), Dst);

            Vec3f _, BaseTranslation;
            solvePnP(Src, Dst, Camera, Mat(), _, BaseTranslation);
            perspectiveTransform(Dst, Dst, Affine);
            solvePnP(Src, Dst, Camera, Mat(), _, RawTranslation);
            RawTranslation -= BaseTranslation;
            if (norm(RawTranslation) <= ArbitraryParameter6) RawTranslation = Vec3f();
            RawTranslation = normalize(RawTranslation);
            RawTranslation[0] *= -1;
            RawTranslation[2] *= -1;

            float Mag = norm(Camera.inv() * Affine * Camera -
                Mat::eye(3, 3, CV_32F)) * ArbitraryParameter3;
            RawTranslation *= Mag;

            if (norm(RawTranslation) >= ArbitraryParameter4) RawTranslation = normalize(RawTranslation) * ArbitraryParameter4;
            else if (norm(RawTranslation) <= ArbitraryParameter6) RawTranslation = Vec3f();

            RawTranslation = (Vec3f)(Mat)(CurrRotation *
                Vec3f(RawTranslation[0], -RawTranslation[1], RawTranslation[2]));
            RawTranslation[1] *= -1;
        }
        catch (...) { ; }

        CurrTranslation += (RawTranslation - CurrTranslation) * ArbitraryParameter7;
        if (norm(RawTranslation) <= ArbitraryParameter6 || norm(CurrTranslation) <= ArbitraryParameter6) {
            RawTranslation = Vec3f();
            CurrTranslation = Vec3f();
        }

        PrevImage = CurrImage.clone();
        PrevRotation = CurrRotation.clone();
        return CurrTranslation;
    }

    Vec4f QuaternionFromRotation(const Mat& rotation) {
        auto quaternion = Quatf::createFromRotMat(rotation);
        return Vec4f(-quaternion.x, +quaternion.y, -quaternion.z, +quaternion.w);
    }

private:
    Ptr<ORB> FeatureDetector = ORB::create();
    BFMatcher FeatureMatcher;

    Mat PrevImage;
    Mat PrevRotation;

    const float ArbitraryParameter1 = 15;
    const float ArbitraryParameter2 = 0.15;
    const float ArbitraryParameter3 = 1.25;
    const float ArbitraryParameter4 = 0.01;
    const float ArbitraryParameter5 = 0.275;
    const float ArbitraryParameter6 = 0.000015;
    const float ArbitraryParameter7 = 0.9;

    Vec3f CurrTranslation;

    void FindMatches(vector<DMatch>& Matches,
        vector<KeyPoint> Keypoints1, Mat Descriptors1,
        vector<KeyPoint> Keypoints2, Mat Descriptors2) {

        if (Keypoints1.empty() || Keypoints2.empty()) return;

        FeatureMatcher.match(Descriptors1, Descriptors2, Matches);

        vector<Point2f> Points1, Points2;
        for (const DMatch& Match : Matches) {
            Points1.push_back(Keypoints1[Match.queryIdx].pt);
            Points2.push_back(Keypoints2[Match.trainIdx].pt);
        }

        Mat InlierMask;
        try {
            findHomography(Points1, Points2, InlierMask, USAC_MAGSAC);
        }
        catch (...) { ; }

        if (InlierMask.total() != Points1.size()) {
            Matches.clear();
            return;
        }

        for (int i = InlierMask.total() - 1; i >= 0; i--)
            if (InlierMask.at<uchar>(i) != 1)
                Matches.erase(Matches.begin() + i);

        Points1.clear();
        Points2.clear();

        for (const DMatch& Match : Matches) {
            Points1.push_back(Keypoints1[Match.queryIdx].pt);
            Points2.push_back(Keypoints2[Match.trainIdx].pt);
        }

        if (Points1.size() >= ArbitraryParameter1) {
            Mat Affine = estimateAffinePartial2D(Points1, Points2);
            Affine.convertTo(Affine, CV_32F);

            if (fabs(Affine.at<float>(0, 2)) >= Resolution * ArbitraryParameter2 ||
                fabs(Affine.at<float>(1, 2)) >= Resolution * ArbitraryParameter2 ||
                fabs(Affine.at<float>(0, 0)) >= 1 + ArbitraryParameter2 ||
                fabs(Affine.at<float>(0, 0)) <= 1 - ArbitraryParameter2 ||
                fabs(Affine.at<float>(1, 1)) >= 1 + ArbitraryParameter2 ||
                fabs(Affine.at<float>(1, 1)) <= 1 - ArbitraryParameter2) Matches.clear();
        }
        else Matches.clear();
    }
};