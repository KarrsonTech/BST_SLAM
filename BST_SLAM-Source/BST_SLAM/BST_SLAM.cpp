/** @file BST_SLAM.cpp
 *  @brief Main XR loop implementation and related logic.
 *
 *  This file contains the core XR loop function which ties together stereo
 *  feature extraction, point cloud generation, and global pose estimation
 *  using IMU rotations.
 */

#include "BST_SLAM.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

 /** @brief Maximum allowable translation between frames. */
static const float MAX_RELATIVE_TRANSLATION = 0.1f;

/** @brief Global pointer to the BST_CAM. */
static BST_CAM* gCamera = new BST_CAM();

/** @brief Global storage of stereo features (left, right, merged) across frames. */
static BST_SLAM::Features gPrevLeftFeatures, gPrevRightFeatures;
static BST_SLAM::Features gPrevMergedFeatures;

/** @brief Global 3D point cloud from the previous frame. */
static std::vector<cv::Point3f> gPrevPointCloud;

/** @brief Global copy of the previous IMU rotation matrix. */
static cv::Mat gPrevIMURotation;

/** @brief Global rotation (Rvec) and translation (Tvec) vectors. */
static cv::Vec3f gGlobalRvec, gGlobalTvec;

/**
    * @brief Forward declaration for relative smoothing function.
    * @param inRvec The current rotation vector.
    * @param inTvec The current translation vector.
    * @param outRvec The smoothed rotation vector.
    * @param outTvec The smoothed translation vector.
*/
void SmoothRelativeObjectPose(const cv::Vec3f& inRvec, const cv::Vec3f& inTvec,
    cv::Vec3f& outRvec, cv::Vec3f& outTvec);

/**
    * @brief Forward declaration for global smoothing function.
    * @param inRvec The current rotation vector.
    * @param inTvec The current translation vector.
    * @param outRvec The smoothed rotation vector.
    * @param outTvec The smoothed translation vector.
*/
void SmoothGlobalObjectPose(const cv::Vec3f& inRvec, const cv::Vec3f& inTvec,
    cv::Vec3f& outRvec, cv::Vec3f& outTvec);

/**
 * @brief Forward declaration for global smoothing function.
 * @param inRvec The current rotation vector.
 * @param inTvec The current translation vector.
 * @param outRvec The smoothed rotation vector.
 * @param outTvec The smoothed translation vector.
 */
void SmoothGlobalObjectPose(const cv::Vec3f& inRvec, const cv::Vec3f& inTvec,
    cv::Vec3f& outRvec, cv::Vec3f& outTvec);

/**
 * @brief Main XR loop. This function is called regularly to update tracking,
 *        compute pose, and visualize results.
 *
 * Retrieves BST_CAM data, updates global pose by comparing current frame
 * against previous frame data (features, IMU rotation, 3D points), and
 * finally visualizes the axes in the debug window.
 */
int main()
{
    using namespace BST_SLAM;

    while (true) {
        // Retrieve BST_CAM data
        BST_CAM::InputData currentData = gCamera->GetInputData();
        cv::Mat currentIMURotation = currentData.Rotation.clone();

        // Build camera intrinsic matrix
        cv::Mat cameraMat = cv::Mat::eye(3, 3, CV_32F);
        cameraMat.at<float>(0, 0) = currentData.fx;
        cameraMat.at<float>(1, 1) = currentData.fy;
        cameraMat.at<float>(0, 2) = currentData.Left.cols * 0.5f;
        cameraMat.at<float>(1, 2) = currentData.Left.rows * 0.5f;

        // If first time, store initial rotation
        if (gPrevIMURotation.empty()) {
            gPrevIMURotation = currentIMURotation.clone();
        }

        // Compute relative rotation
        cv::Mat relRotation = (gPrevIMURotation.inv() * currentIMURotation).inv();

        // Basic check to see if IMU rotation isn’t changing too much
        float rotationDiff = static_cast<float>(cv::norm(relRotation - cv::Mat::eye(3, 3, CV_32F)));
        cv::Vec3f relRvec, relTvec(0, 0, 0);

        // Prepare features for current frame
        Features currentMergedFeatures;
        Features leftFeatures = GetFeatures(currentData.Left);
        Features rightFeatures = GetFeatures(currentData.Right);

        static cv::Mat gPrevIMURotation2 = currentIMURotation.clone();

        // Only do complex logic if rotation changed within a small threshold
        if (rotationDiff < 0.15f)
        {
            try
            {
                gPrevIMURotation2 = currentIMURotation.clone();

                // Generate 3D point cloud from stereo
                std::vector<cv::Point3f> currentPointCloud = GetPointCloud(currentData,
                    leftFeatures,
                    rightFeatures,
                    currentMergedFeatures);

                bool gotCurrentCloud = !currentPointCloud.empty();

                // If no valid data this frame, try again with previous stereo features
                if (!gotCurrentCloud)
                {
                    leftFeatures = gPrevLeftFeatures;
                    rightFeatures = gPrevRightFeatures;
                    currentPointCloud = GetPointCloud(currentData,
                        leftFeatures,
                        rightFeatures,
                        currentMergedFeatures);
                }

                if (!currentPointCloud.empty())
                {
                    // If first time we have a cloud, set previous
                    if (gPrevPointCloud.empty())
                    {
                        gPrevMergedFeatures = currentMergedFeatures;
                        gPrevPointCloud = currentPointCloud;
                    }

                    // Match current and previous features to get 2D-2D correspondences
                    std::vector<cv::DMatch> matches;
                    GetMatches(matches, gPrevMergedFeatures, currentMergedFeatures);

                    // Re-check rotation
                    currentIMURotation.convertTo(currentIMURotation, CV_32F);
                    if (gPrevIMURotation.empty()) {
                        gPrevIMURotation = currentIMURotation.clone();
                    }
                    relRotation = (gPrevIMURotation.inv() * currentIMURotation).inv();

                    if (!matches.empty())
                    {
                        // Collect matched keypoints from the previous & current frames
                        std::vector<cv::Point2f> prev2D, curr2D;
                        std::vector<cv::Point3f> prev3D;
                        prev2D.reserve(matches.size());
                        curr2D.reserve(matches.size());
                        prev3D.reserve(matches.size());

                        for (auto& m : matches)
                        {
                            prev2D.push_back(gPrevMergedFeatures.Keypoints[m.queryIdx].pt);
                            curr2D.push_back(currentMergedFeatures.Keypoints[m.trainIdx].pt);
                            prev3D.push_back(gPrevPointCloud[m.queryIdx]);
                        }

                        // Estimate partial affine
                        cv::Mat tmpAffine = cv::estimateAffinePartial2D(prev2D, curr2D);
                        tmpAffine.convertTo(tmpAffine, CV_32F);
                        cv::vconcat(tmpAffine, cv::Vec3f(0, 0, 1).t(), tmpAffine);

                        // Combine with the camera matrix + relative IMU rotation
                        cv::Mat relHomography = (cameraMat * relRotation * cameraMat.inv()) * tmpAffine;

                        // Use the new homography to project points
                        std::vector<cv::Point2f> projectedPrev2D;
                        cv::projectPoints(prev3D, cv::Vec3f(), cv::Vec3f(), cameraMat, cv::noArray(), projectedPrev2D);
                        cv::perspectiveTransform(projectedPrev2D, projectedPrev2D, relHomography);

                        // Solve PnP for relative motion
                        cv::solvePnP(prev3D, projectedPrev2D, cameraMat, cv::noArray(), relRvec, relTvec);
                    }

                    // Apply perspective transform to left & right images so next point cloud is consistent
                    if (!leftFeatures.Keypoints.empty() || !rightFeatures.Keypoints.empty())
                    {
                        std::vector<cv::Point2f> lPts, rPts;
                        lPts.reserve(leftFeatures.Keypoints.size());
                        rPts.reserve(rightFeatures.Keypoints.size());

                        for (auto& kp : leftFeatures.Keypoints)  lPts.push_back(kp.pt);
                        for (auto& kp : rightFeatures.Keypoints) rPts.push_back(kp.pt);

                        cv::perspectiveTransform(lPts, lPts, cameraMat * relRotation * cameraMat.inv());
                        cv::perspectiveTransform(rPts, rPts, cameraMat * relRotation * cameraMat.inv());

                        for (size_t i = 0; i < leftFeatures.Keypoints.size(); i++) {
                            leftFeatures.Keypoints[i].pt = lPts[i];
                        }
                        for (size_t i = 0; i < rightFeatures.Keypoints.size(); i++) {
                            rightFeatures.Keypoints[i].pt = rPts[i];
                        }

                        // Recompute the point cloud with warped images
                        currentPointCloud = GetPointCloud(currentData,
                            leftFeatures,
                            rightFeatures,
                            currentMergedFeatures);
                    }

                    if (gotCurrentCloud)
                    {
                        gPrevPointCloud = currentPointCloud;
                    }
                }

                // If translation is within a reasonable range, compose global motion
                float magRelTvec = cv::norm(relTvec);
                if (magRelTvec > 0 && magRelTvec < MAX_RELATIVE_TRANSLATION)
                {
                    cv::Vec3f relRvecSmooth, relTvecSmooth;
                    SmoothRelativeObjectPose(relRvec, relTvec,
                        relRvecSmooth, relTvecSmooth);

                    cv::composeRT(gGlobalRvec, gGlobalTvec,
                        relRvecSmooth, relTvecSmooth,
                        gGlobalRvec, gGlobalTvec);
                }
            }
            catch (...)
            {
                // Fallback to rotation-only logic if something goes wrong
                // (handled outside the try block)
                goto rotation_only;
            }
        }
        else
        {
        rotation_only:
            if (gPrevPointCloud.size() >= 6)
            {
                // Project points from the previous cloud
                std::vector<cv::Point2f> imagePts;
                cv::projectPoints(gPrevPointCloud, cv::Vec3f(), cv::Vec3f(), cameraMat, cv::noArray(), imagePts);

                if (gPrevIMURotation2.empty()) {
                    gPrevIMURotation2 = currentIMURotation.clone();
                }
                cv::Mat relRotation2 = (gPrevIMURotation2.inv() * currentIMURotation).inv();
                cv::perspectiveTransform(imagePts, imagePts, cameraMat * relRotation2 * cameraMat.inv());

                cv::solvePnP(gPrevPointCloud, imagePts, cameraMat, cv::noArray(),
                    relRvec, relTvec);

                cv::composeRT(gGlobalRvec, gGlobalTvec,
                    relRvec, relTvec,
                    gGlobalRvec, gGlobalTvec);
            }
        }

        cv::Vec3f gGlobalRvec, gGlobalTvec;
        SmoothGlobalObjectPose(::gGlobalRvec, ::gGlobalTvec,
            gGlobalRvec, gGlobalTvec);

        // Store everything for next iteration
        gPrevLeftFeatures = leftFeatures;
        gPrevRightFeatures = rightFeatures;
        gPrevMergedFeatures = currentMergedFeatures;
        gPrevIMURotation = currentIMURotation.clone();

        // Visualization
        cv::Mat debugVis = currentData.Left.clone();
        cv::drawFrameAxes(debugVis, cameraMat, cv::noArray(),
            gGlobalRvec, gGlobalTvec, 0.3);

        cv::imshow("Vis", debugVis);
        cv::waitKey(1);
    }

    return 0;
}

/**
 * @brief Interpolates the relative object pose using hard smoothing.
 * @param inRvec The current rotation vector.
 * @param inTvec The current translation vector.
 * @param outRvec The smoothed rotation vector.
 * @param outTvec The smoothed translation vector.
 */
void SmoothRelativeObjectPose(const cv::Vec3f& inRvec, const cv::Vec3f& inTvec,
    cv::Vec3f& outRvec, cv::Vec3f& outTvec)
{
    int value = 2;
    static std::vector<cv::Vec3f> rvecSlidingWindow;
    static std::vector<cv::Vec3f> tvecSlidingWindow;
    rvecSlidingWindow.push_back(inRvec);
    tvecSlidingWindow.push_back(inTvec);
    outRvec = cv::Vec3f();
    outTvec = cv::Vec3f();
    for (const auto& rvec : rvecSlidingWindow) outRvec += rvec;
    for (const auto& tvec : tvecSlidingWindow) outTvec += tvec;
    outRvec /= (float)rvecSlidingWindow.size();
    outTvec /= (float)tvecSlidingWindow.size();
    if (rvecSlidingWindow.size() >= value) {
        rvecSlidingWindow.erase(rvecSlidingWindow.begin());
        tvecSlidingWindow.erase(tvecSlidingWindow.begin());
    }
}

/**
 * @brief Interpolates the global object pose using soft smoothing.
 * @param inRvec The current rotation vector.
 * @param inTvec The current translation vector.
 * @param outRvec The smoothed rotation vector.
 * @param outTvec The smoothed translation vector.
 */
void SmoothGlobalObjectPose(const cv::Vec3f& inRvec, const cv::Vec3f& inTvec, 
    cv::Vec3f& outRvec, cv::Vec3f& outTvec)
{
    float value = 0.75f;
    static cv::Vec3f smoothedRvec = inRvec;
    static cv::Vec3f smoothedTvec = inTvec;
    smoothedRvec += (inRvec - smoothedRvec) * value;
    smoothedTvec += (inTvec - smoothedTvec) * value;
    outRvec = smoothedRvec;
    outTvec = smoothedTvec;
}