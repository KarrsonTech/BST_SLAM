#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>
#include <vector>
#include "BST_VIO.hpp"

namespace BST_SLAM {
    static void VerifyRotation(cv::Mat Rotation, cv::Mat Image) {
        BST_VIO::VerifyRotation(Rotation, Image);
    }

    static cv::Vec4f QuaternionFromRotation(const cv::Mat& Rotation) {
        return BST_VIO::QuaternionFromRotation(Rotation);
    }

    static cv::Vec3f EstimatePosition3D(cv::Mat CurrentFrame, cv::Mat CurrentRotation) {
        constexpr float MIN_PIX_THRESHOLD = 0.25f;
        constexpr float MAX_NUM_MAP_NODES = 250;
        constexpr float NODE_TIMER_THRESH = 0.5;
        constexpr float TIME_RELOC_FACTOR = 0.25f;
        constexpr float PIX2MET_SMOOTHING = 0.75f;
        constexpr float SPATIAL_SMOOTHING = 49.375f;
        constexpr float SLOWER_FRAME_RATE = 50.0f;

        struct MapNode {
            cv::Mat image;
            cv::Vec3f position;
            cv::Mat rotation;
            clock_t timestamp;
        };

        static cv::Vec3f position(0, 0, 0); 
        static std::vector<MapNode> mapNodes;
        static cv::Mat prevImage;
        static clock_t lastNodeTime = clock();
        static clock_t prevTime = clock();
        static float pixelToMetricRatio = 1.0f;
        static cv::Vec3f rawPosition(0, 0, 0);
        static size_t currentNodeIndex = 0;

        auto IsFrameCloseEnough = [&](const cv::Mat& current, const cv::Mat& reference, cv::Mat& H) {
            cv::Mat currentResized = current.clone();
            cv::Mat referenceResized = reference.clone();

            const int targetSize = BST_VIO::detail::IMAGE_SIZE;
            cv::resize(currentResized, currentResized, cv::Size(targetSize, targetSize));
            cv::resize(referenceResized, referenceResized, cv::Size(targetSize, targetSize));

            H = BST_VIO::detail::detectAndMatchFeatures(referenceResized, currentResized);
            if (H.empty()) return false;

            H.convertTo(H, CV_32F);
            float pixelShift = cv::norm(H - cv::Mat::eye(3, 3, CV_32F));
            return pixelShift > FLT_EPSILON && pixelShift < MIN_PIX_THRESHOLD * currentResized.cols;
            };

        auto UpdatePixelToMetricRatio = [&](const cv::Vec3f& physicalMotion, const cv::Mat& H) {
            float pixelMotion = cv::norm(H - cv::Mat::eye(3, 3, CV_32F));
            if (pixelMotion > 0) {
                float newRatio = cv::norm(physicalMotion) / pixelMotion;
                pixelToMetricRatio = PIX2MET_SMOOTHING * pixelToMetricRatio + (1.0f - PIX2MET_SMOOTHING) * newRatio;
            }
            };

        auto Lerp = [&](cv::Vec3f v0, cv::Vec3f v1, float t) {
            return (1 - t) * v0 + t * v1;
            };

        auto ProcessLoopClosure = [&](const cv::Mat& currentFrame,
            const cv::Vec3f& currentPosition,
            const cv::Vec3f& vioMotion) -> cv::Vec3f {
                if (mapNodes.empty()) return currentPosition;

                currentNodeIndex %= mapNodes.size();
                const MapNode& node = mapNodes[currentNodeIndex++];

                float deltaTime = float(clock() - prevTime) / CLOCKS_PER_SEC;
                prevTime = clock();

                cv::Mat H;
                if (IsFrameCloseEnough(currentFrame, node.image, H) && deltaTime > 0) {
                    float pixelMotion = cv::norm(H - cv::Mat::eye(3, 3, CV_32F));
                    cv::Vec3f estimatedMotion = vioMotion * (pixelMotion * pixelToMetricRatio);
                    cv::Vec3f relocTarget = node.position + estimatedMotion;

                    return Lerp(currentPosition, relocTarget, TIME_RELOC_FACTOR * deltaTime);
                }

                return currentPosition;
            };

        cv::Vec3f vioTranslation = BST_VIO::EstimateTranslation3D(CurrentFrame, CurrentRotation);
        rawPosition += vioTranslation;

        clock_t currentTime = clock();
        float mapNodeTimeDelta = float(currentTime - lastNodeTime) / CLOCKS_PER_SEC;

        if (!prevImage.empty()) {
            cv::Mat H;
            if (!mapNodes.empty() && IsFrameCloseEnough(CurrentFrame, mapNodes[mapNodes.size() - 1].image, H)) {
                UpdatePixelToMetricRatio(vioTranslation, H);
            }
            else if (mapNodes.empty() || mapNodeTimeDelta >= NODE_TIMER_THRESH) {
                MapNode newNode{
                    CurrentFrame.clone(),
                    rawPosition,
                    CurrentRotation.clone(),
                    currentTime
                };

                mapNodes.push_back(newNode);
                lastNodeTime = currentTime;
            }
        }
        while (mapNodes.size() > MAX_NUM_MAP_NODES) mapNodes.erase(mapNodes.begin());

        prevImage = CurrentFrame.clone();
        rawPosition = ProcessLoopClosure(CurrentFrame, rawPosition, vioTranslation);

        float deltaTime = float(clock() - prevTime) / CLOCKS_PER_SEC;
        while (deltaTime < 1.0f / SLOWER_FRAME_RATE) deltaTime = float(clock() - prevTime) / CLOCKS_PER_SEC;
        prevTime = clock();
        static float Mix = 1.0f;
        float RawMix = SPATIAL_SMOOTHING * deltaTime;
        if (RawMix < 1.0f) Mix = RawMix;
        position = Lerp(position, rawPosition, Mix);
        return position;
    }
}