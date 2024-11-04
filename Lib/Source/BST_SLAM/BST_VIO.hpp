#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>

namespace BST_VIO {
    static void VerifyRotation(cv::Mat Rotation, cv::Mat Image) {
        Rotation = Rotation.clone();
        Image = Image.clone();

        cv::Mat debugImage;
        cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        cameraMatrix.at<double>(0, 0) = cameraMatrix.at<double>(1, 1) = 800;
        cameraMatrix.at<double>(0, 2) = cameraMatrix.at<double>(1, 2) = 400;

        cv::Mat rotationDouble = Rotation.clone();
        rotationDouble.convertTo(rotationDouble, CV_64F);

        cv::resize(Image, Image, cv::Size(800, 800));
        cv::warpPerspective(Image, debugImage,
            cameraMatrix * rotationDouble * cameraMatrix.inv(),
            Image.size());

        cv::imshow("The rotation should stabilize to zero. Please check.", debugImage);
    }

    static cv::Vec4f QuaternionFromRotation(const cv::Mat& Rotation) {
        auto quaternion = cv::Quatf::createFromRotMat(Rotation);
        return cv::Vec4f(-quaternion.x, +quaternion.y, -quaternion.z, +quaternion.w);
    }

    cv::Vec3f EstimateTranslation3D(cv::Mat CurrentFrame, cv::Mat CurrentRotation);

    namespace detail {
        // Core tracking configuration
        constexpr int FEATURE_COUNT = 760;
        constexpr float MATCH_RATIO = 0.85f;
        constexpr float MIN_CONFIDENCE = 0.3f;
        constexpr double SHAKE_SENSITIVITY = 1.25;
        constexpr double AVG_VELOCITY = 0.01;
        constexpr int IMAGE_SIZE = 360;
        constexpr float ROTATION_THRESHOLD = 2.0f;
        constexpr int WARMUP_FRAMES = 10;
        constexpr double DIRECTION_EPSILON = 0.01;
        constexpr double SMOOTH_FACTOR = 0.95;

        // Global state
        static cv::Vec3f worldPosition(0, 0, 0);
        static cv::Mat previousFrame;
        static cv::Mat previousRotation;
        static cv::Mat initialTransform;
        static cv::Vec3d movementDirection;
        static cv::Vec3f smoothedTranslation;
        static cv::Vec3f velocity;
        static int frameCount = 0;
        static clock_t lastTimestamp = clock();

        // Feature detection
        static cv::Ptr<cv::ORB> orbDetector = cv::ORB::create(FEATURE_COUNT);
        static cv::Ptr<cv::BFMatcher> featureMatcher = cv::BFMatcher::create();

        cv::Mat detectAndMatchFeatures(const cv::Mat& prevImage, const cv::Mat& currentImage) {
            std::vector<cv::KeyPoint> prevKeypoints, currentKeypoints;
            cv::Mat prevDescriptors, currentDescriptors;
            std::vector<cv::Point2f> sourcePoints, targetPoints;

            orbDetector->detectAndCompute(prevImage, cv::noArray(), prevKeypoints, prevDescriptors);
            orbDetector->detectAndCompute(currentImage, cv::noArray(), currentKeypoints, currentDescriptors);

            if (prevKeypoints.size() < 4 || currentKeypoints.size() < 4) {
                return cv::Mat::eye(3, 3, CV_64F);
            }

            std::vector<std::vector<cv::DMatch>> matches;
            featureMatcher->knnMatch(prevDescriptors, currentDescriptors, matches, 2);

            for (const auto& match : matches) {
                if (match.size() >= 2 && match[0].distance <= MATCH_RATIO * match[1].distance) {
                    sourcePoints.push_back(cv::Point2f(
                        (double)prevKeypoints[match[0].queryIdx].pt.x,
                        (double)prevKeypoints[match[0].queryIdx].pt.y
                    ));
                    targetPoints.push_back(cv::Point2f(
                        (double)currentKeypoints[match[0].trainIdx].pt.x,
                        (double)currentKeypoints[match[0].trainIdx].pt.y
                    ));
                }
            }

            if (sourcePoints.size() < 4) {
                return cv::Mat::eye(3, 3, CV_64F);
            }

            cv::Mat inlierMask;
            cv::Mat homography = cv::findHomography(sourcePoints, targetPoints, inlierMask,
                sourcePoints.size() >= 15 ? cv::USAC_MAGSAC : cv::RANSAC);
            inlierMask.convertTo(inlierMask, CV_8U);

            int originalCount = sourcePoints.size();
            for (int i = inlierMask.rows - 1; i >= 0; i--) {
                if ((int)inlierMask.at<uchar>(i) == 0) {
                    sourcePoints.erase(sourcePoints.begin() + i);
                    targetPoints.erase(targetPoints.begin() + i);
                }
            }

            if (sourcePoints.size() >= 3) {
                cv::Mat affine = cv::estimateAffinePartial2D(sourcePoints, targetPoints);
                affine.convertTo(affine, CV_64F);
                float confidence = (double)sourcePoints.size() / (8.0f + 0.3f * (double)originalCount);
                if (!affine.empty() && confidence >= MIN_CONFIDENCE) {
                    cv::Mat fullHomography;
                    cv::vconcat(affine, cv::Vec3d(0, 0, 1).t(), fullHomography);
                    return fullHomography;
                }
            }

            return cv::Mat::eye(3, 3, CV_64F);
        }

        cv::Mat estimateTransformation(const cv::Mat& homography, const cv::Mat& cameraMatrix) {
            cv::Mat transformation = cv::Mat::eye(3, 3, CV_64F);

            // Calculate transformation for center point
            std::vector<cv::Point2f> centerPoint{ cv::Point2f(IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0) };
            std::vector<cv::Point2f> transformedPoint;
            cv::perspectiveTransform(centerPoint, transformedPoint, homography);

            transformation.at<double>(0, 2) = transformedPoint[0].x - centerPoint[0].x;
            transformation.at<double>(1, 2) = transformedPoint[0].y - centerPoint[0].y;
            double scale = homography.at<double>(0, 0);
            transformation.at<double>(0, 0) = transformation.at<double>(1, 1) = scale;

            transformation.at<double>(0, 2) -= (scale - 1.0) * centerPoint[0].x;
            transformation.at<double>(1, 2) -= (scale - 1.0) * centerPoint[0].y;

            // Calculate full transformation using PnP
            std::vector<cv::Point3d> objectPoints = {
                cv::Point3d(-0.5, 0.5, 0), cv::Point3d(0.5, 0.5, 0),
                cv::Point3d(0.5, -0.5, 0), cv::Point3d(-0.5, -0.5, 0)
            };
            std::vector<cv::Point2d> imagePoints = {
                cv::Point2d(0, 0), cv::Point2d(IMAGE_SIZE, 0),
                cv::Point2d(IMAGE_SIZE, IMAGE_SIZE), cv::Point2d(0, IMAGE_SIZE)
            };

            std::vector<cv::Point2d> projectedPoints;
            cv::Mat tempTransform = transformation.clone();
            cv::perspectiveTransform(imagePoints, projectedPoints, transformation);

            cv::Vec3d rotationVec;
            cv::solvePnP(objectPoints, projectedPoints, cameraMatrix, cv::Mat(), rotationVec, transformation);
            transformation.convertTo(transformation, CV_64F);

            // Initialize reference transform if needed
            if (initialTransform.empty()) {
                initialTransform = transformation.clone();
            }

            transformation -= initialTransform;
            transformation.at<double>(0) *= -1;
            transformation.at<double>(2) *= -1;

            return transformation;
        }

        cv::Vec3f calculateMovement(const cv::Mat& transform, const cv::Mat& cameraMatrix,
            const cv::Mat& tempTransform, const cv::Mat& rotation) {
            cv::Vec3d translation(transform.at<double>(0), transform.at<double>(1), transform.at<double>(2));

            // Update movement direction
            if (cv::norm(translation) > 0) {
                translation /= cv::norm(translation);
            }
            else {
                translation = movementDirection;
            }

            if (std::abs(cv::norm(translation) - 1) <= DIRECTION_EPSILON) {
                movementDirection = translation;
            }
            else {
                translation = movementDirection;
            }

            // Scale translation based on transformation
            double transformMagnitude = cv::norm(cameraMatrix.inv() * tempTransform * cameraMatrix -
                cv::Mat::eye(3, 3, CV_64F)) * SHAKE_SENSITIVITY;
            translation *= transformMagnitude;

            // Apply velocity constraint
            if (cv::norm(translation) >= AVG_VELOCITY) {
                translation = movementDirection * AVG_VELOCITY;
            }

            // Convert to world coordinates
            cv::Vec3f worldTranslation = (cv::Vec3f)(cv::Mat)(rotation *
                cv::Vec3f(translation[0], -translation[1], translation[2]));
            worldTranslation[1] *= -1;

            return worldTranslation;
        }

        static cv::Vec3f EstimatePosition3D(cv::Mat currentFrame, cv::Mat currentRotation) {
            currentFrame = currentFrame.clone();
            currentRotation = currentRotation.clone();

            try {
                clock_t currentTime = clock();

                // Initialize if needed
                if (previousFrame.empty()) previousFrame = currentFrame;
                if (previousRotation.empty()) previousRotation = currentRotation;

                // Resize frames
                cv::resize(previousFrame, previousFrame, cv::Size(IMAGE_SIZE, IMAGE_SIZE), 0, 0, cv::INTER_AREA);
                cv::resize(currentFrame, currentFrame, cv::Size(IMAGE_SIZE, IMAGE_SIZE), 0, 0, cv::INTER_AREA);

                // Setup transformations
                cv::Mat rotation1 = currentRotation.clone();
                cv::Mat rotation2 = (previousRotation.inv() * currentRotation).inv();
                rotation1.convertTo(rotation1, CV_32F);
                rotation2.convertTo(rotation2, CV_64F);

                // Setup camera matrix
                cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
                cameraMatrix.at<double>(0, 0) = cameraMatrix.at<double>(1, 1) = IMAGE_SIZE;
                cameraMatrix.at<double>(0, 2) = cameraMatrix.at<double>(1, 2) = IMAGE_SIZE / 2.0;

                // Compensate rotation in previous frame
                cv::Mat rotationTransform = (previousRotation.inv() * currentRotation).inv();
                rotationTransform.convertTo(rotationTransform, CV_64F);
                rotationTransform = cameraMatrix * rotationTransform * cameraMatrix.inv();
                cv::warpPerspective(previousFrame, previousFrame, rotationTransform, currentFrame.size());

                // Match features and estimate homography
                cv::Mat homography = detectAndMatchFeatures(previousFrame, currentFrame);
                cv::Mat transform = estimateTransformation(homography, cameraMatrix);
                cv::Vec3f translation = calculateMovement(transform, cameraMatrix, homography, rotation1);

                // Time-based smoothing
                float deltaTime = (float)(currentTime - lastTimestamp) / (float)CLOCKS_PER_SEC;
                float rotationMagnitude = (float)cv::norm(rotation2 - cv::Mat::eye(3, 3, CV_64F)) / deltaTime;
                float blendFactor = std::max(0.0f, (ROTATION_THRESHOLD - rotationMagnitude)) / ROTATION_THRESHOLD;

                smoothedTranslation = translation * blendFactor + smoothedTranslation * (1.0f - blendFactor);
                translation = smoothedTranslation;

                // Update position after warmup
                if (frameCount >= WARMUP_FRAMES) {
                    velocity += translation * deltaTime;
                    translation = velocity * deltaTime * (1.0f - SMOOTH_FACTOR) + translation * SMOOTH_FACTOR;
                    velocity = translation / deltaTime;

                    worldPosition += translation;
                }
                else {
                    frameCount++;
                }

                // Store state for next frame
                previousFrame = currentFrame.clone();
                previousRotation = currentRotation.clone();
                lastTimestamp = currentTime;

                return worldPosition;
            }
            catch (...) {
                previousFrame = currentFrame.clone();
                previousRotation = currentRotation.clone();
                return worldPosition;
            }
        }
    }

    static cv::Vec3f EstimateTranslation3D(cv::Mat CurrentFrame, cv::Mat CurrentRotation) {
        static cv::Vec3f tvecVIO(0, 0, 0);
        {
            cv::Vec3f currentPositionVIO = BST_VIO::detail::EstimatePosition3D(CurrentFrame, CurrentRotation);
            static cv::Vec3f previousPositionVIO = currentPositionVIO;
            tvecVIO = currentPositionVIO - previousPositionVIO;
            previousPositionVIO = currentPositionVIO;
        }
        return tvecVIO;
    }
}