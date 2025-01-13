/**
 * @file BST_SLAM.hpp
 * @brief Dependencies for BST_SLAM.cpp
 */

#pragma once
#include <BST_CAM.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>

// Feature detection and matching utilities using ORB and BFMatcher.

namespace BST_SLAM
{
    /**
     * @brief ORB detector with a maximum of 750 features.
     * @details This is a global static pointer used for feature detection.
     */
    static cv::Ptr<cv::ORB> FeatureDetector = cv::ORB::create(750);

    /**
     * @brief Brute force feature matcher (cross-check disabled by default).
     * @details Used to match feature descriptors between frames.
     */
    static cv::BFMatcher    FeatureMatcher;

    /**
     * @struct Features
     * @brief Contains keypoints and descriptors for a single frame.
     */
    struct Features
    {
        /** @brief Vector of detected ORB keypoints. */
        std::vector<cv::KeyPoint> Keypoints;
        /** @brief Mat of corresponding descriptors. */
        cv::Mat Descriptors;
    };

    /**
     * @brief Extract ORB features from an image.
     * @param imageInput The input image from which to detect features.
     * @return A Features struct containing detected keypoints and descriptors.
     */
    inline Features GetFeatures(const cv::Mat& imageInput)
    {
        Features features;
        cv::Mat imageGray = (imageInput.channels() != 1)
            ? [&] {
            cv::Mat tmp;
            cv::cvtColor(imageInput, tmp, cv::COLOR_BGR2GRAY);
            return tmp;
            }()
                : imageInput.clone();

            FeatureDetector->detectAndCompute(imageGray, cv::noArray(),
                features.Keypoints,
                features.Descriptors);
            return features;
    }

    /**
     * @brief Append features from multiple sources into a single Features object.
     * @param sources A vector of Features objects to be merged.
     * @param target The target Features object to which new features are added.
     */
    inline void AddFeatures(const std::vector<Features>& sources, Features& target)
    {
        for (const auto& fs : sources)
        {
            // Append keypoints
            target.Keypoints.insert(target.Keypoints.end(),
                fs.Keypoints.begin(),
                fs.Keypoints.end());
            // Concatenate descriptors
            if (!fs.Descriptors.empty())
            {
                if (target.Descriptors.empty())
                {
                    target.Descriptors = fs.Descriptors.clone();
                }
                else
                {
                    cv::vconcat(target.Descriptors, fs.Descriptors, target.Descriptors);
                }
            }
        }
    }

    /**
     * @brief Overload to append a single Features object to the target.
     * @param single A single Features object to be added.
     * @param target The target Features object to which the new features are added.
     */
    inline void AddFeatures(const Features& single, Features& target)
    {
        AddFeatures(std::vector<Features>{ single }, target);
    }

    /**
     * @brief Use BFMatcher to match features between two feature sets.
     * @param matches [out] The resulting set of matches.
     * @param f1 Features from the first image.
     * @param f2 Features from the second image.
     * @param skipLowesRatio Whether to skip Lowe's ratio test (default false).
     * @details The function uses a knnMatch and applies a ratio test,
     *          followed by homography-based outlier filtering with MAGSAC.
     */
    inline void GetMatches(std::vector<cv::DMatch>& matches,
        const Features& f1,
        const Features& f2,
        bool skipLowesRatio = false)
    {
        matches.clear();
        if (f1.Keypoints.empty() || f2.Keypoints.empty()) return;

        std::vector<std::vector<cv::DMatch>> knnMatches;
        FeatureMatcher.knnMatch(f1.Descriptors, f2.Descriptors, knnMatches, 2);

        std::vector<cv::Point2f> pts1, pts2;
        for (auto& m : knnMatches)
        {
            if (m.size() < 2) continue;

            bool passLowe = (m[0].distance <= 0.0f) ||
                (m[0].distance <= m[1].distance * 0.95f) ||
                skipLowesRatio;

            if (passLowe)
            {
                pts1.push_back(f1.Keypoints[m[0].queryIdx].pt);
                pts2.push_back(f2.Keypoints[m[0].trainIdx].pt);
                matches.push_back(m[0]);
            }
        }

        // Homography + MAGSAC to filter outliers
        if (pts1.empty() || pts2.empty()) {
            matches.clear();
            return;
        }

        cv::Mat inlierMask;
        try
        {
            cv::findHomography(pts1, pts2, inlierMask, cv::USAC_MAGSAC);
        }
        catch (...)
        {
            matches.clear();
            return;
        }

        if (inlierMask.empty() ||
            inlierMask.total() != pts1.size() ||
            (cv::countNonZero(inlierMask) / (8 + 0.3f * (float)knnMatches.size())) <= 0.05)
        {
            matches.clear();
            return;
        }

        // Keep only inliers
        for (int i = (int)inlierMask.total() - 1; i >= 0; i--)
        {
            if (inlierMask.at<uchar>(i) == 0) {
                matches.erase(matches.begin() + i);
            }
        }

        // Enforce a minimum number of matches
        if (matches.size() < 6) {
            matches.clear();
        }
    }
}

// Stereo triangulation utilities for generating 3D point clouds.

namespace BST_SLAM
{
    /**
     * @brief Triangulate stereo keypoints into a 3D point cloud.
     * @param inputData The BST_CAM input containing left/right images, IMU rotation, etc.
     * @param leftFeatures ORB features (keypoints + descriptors) from the left image.
     * @param rightFeatures ORB features (keypoints + descriptors) from the right image.
     * @param mergedFeatures [out] A combined Features struct containing merged descriptors.
     * @return A vector of 3D points representing the triangulated stereo cloud.
     *
     * This function matches features between the left and right frames, then
     * uses triangulation (with the provided baseline distance) to compute
     * a 3D point cloud. Each 3D point is then rotated by the IMU rotation matrix.
     * The mergedFeatures will contain the "averaged" 2D keypoints (from left/right)
     * and descriptors (copied from the left side).
     */
    inline std::vector<cv::Point3f> GetPointCloud(
        const BST_CAM::InputData& inputData,
        const Features& leftFeatures,
        const Features& rightFeatures,
        Features& mergedFeatures
    )
    {
        mergedFeatures.Keypoints.clear();
        mergedFeatures.Descriptors.release();

        // Early exit if either descriptor set is empty
        if (leftFeatures.Descriptors.empty() || rightFeatures.Descriptors.empty()) {
            return {};
        }

        // Perform feature matching
        std::vector<cv::DMatch> stereoMatches;
        GetMatches(stereoMatches, leftFeatures, rightFeatures, true);
        if (stereoMatches.empty()) {
            return {};
        }

        // Prepare descriptors for mergedFeatures
        mergedFeatures.Descriptors = cv::Mat::zeros(
            (int)stereoMatches.size(),
            leftFeatures.Descriptors.cols,
            leftFeatures.Descriptors.type()
        );

        // Prepare vectors for triangulation
        std::vector<cv::Point2f> leftPts, rightPts;
        leftPts.reserve(stereoMatches.size());
        rightPts.reserve(stereoMatches.size());

        for (size_t i = 0; i < stereoMatches.size(); i++)
        {
            const auto& m = stereoMatches[i];
            leftPts.push_back(leftFeatures.Keypoints[m.queryIdx].pt);
            rightPts.push_back(rightFeatures.Keypoints[m.trainIdx].pt);
        }

        // Intrinsic camera matrix
        cv::Mat cameraMat = cv::Mat::eye(3, 3, CV_32F);
        cameraMat.at<float>(0, 0) = inputData.fx;
        cameraMat.at<float>(1, 1) = inputData.fy;
        cameraMat.at<float>(0, 2) = inputData.Left.cols * 0.5f;
        cameraMat.at<float>(1, 2) = inputData.Left.rows * 0.5f;

        // Projection for left camera
        cv::Mat P1 = cv::Mat::eye(3, 4, CV_32F);
        P1.at<float>(0, 3) = inputData.BaselineDistance / 2.0f; // shift on x-axis
        P1 = cameraMat * P1;

        // Projection for right camera
        cv::Mat P2 = cv::Mat::eye(3, 4, CV_32F);
        P2.at<float>(0, 3) = -(inputData.BaselineDistance / 2.0f);
        P2 = cameraMat * P2;

        // Triangulate points
        cv::Mat homogeneousPts;
        cv::triangulatePoints(P1, P2, leftPts, rightPts, homogeneousPts);

        // Convert to 3D
        cv::Mat cartesianPts;
        cv::convertPointsFromHomogeneous(homogeneousPts.t(), cartesianPts);
        cartesianPts.convertTo(cartesianPts, CV_32F);

        // Transform each 3D point with IMU rotation
        std::vector<cv::Point3f> pointCloud;
        pointCloud.reserve(cartesianPts.rows);

        for (int i = 0; i < cartesianPts.rows; i++)
        {
            cv::Point3f pt3D = cartesianPts.at<cv::Point3f>(i);

            // Flip y-axis, apply rotation, flip y-axis again
            pt3D.y *= -1.0f;
            pt3D = (cv::Point3f)cv::Mat(inputData.Rotation * cv::Mat(pt3D));
            pt3D.y *= -1.0f;

            pointCloud.push_back(pt3D);

            // Merge features: average left+right 2D location
            auto leftPt = leftFeatures.Keypoints[stereoMatches[i].queryIdx].pt;
            auto rightPt = rightFeatures.Keypoints[stereoMatches[i].trainIdx].pt;
            cv::Point2f avgPt2D((leftPt.x + rightPt.x) * 0.5f,
                (leftPt.y + rightPt.y) * 0.5f);

            float avgSize = (leftFeatures.Keypoints[stereoMatches[i].queryIdx].size +
                rightFeatures.Keypoints[stereoMatches[i].trainIdx].size) * 0.5f;
            mergedFeatures.Keypoints.emplace_back(avgPt2D, avgSize);

            // Copy the descriptor from the left side
            leftFeatures.Descriptors.row(stereoMatches[i].queryIdx)
                .copyTo(mergedFeatures.Descriptors.row(i));
        }

        // Enforce a minimum size on the 3D set
        if (pointCloud.size() < 6)
        {
            pointCloud.clear();
            mergedFeatures.Keypoints.clear();
            mergedFeatures.Descriptors.release();
        }

        return pointCloud;
    }
}
