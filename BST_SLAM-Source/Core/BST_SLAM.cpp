#include "../User/SensorDriver.hpp"
#include "BST_SLAM/Solver.hpp"
#include "BST_SLAM/Messaging.hpp"
#include <opencv2/ximgproc.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <unordered_map>
#include <cmath>
#include <iostream>

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

struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelHash {
    std::size_t operator()(const VoxelKey& key) const {
        return std::hash<int>()(key.x) ^ std::hash<int>()(key.y) ^ std::hash<int>()(key.z);
    }
};

class OccupancyGrid3D {
public:
    OccupancyGrid3D(float voxel_size = 0.05, float max_depth = 5.0)
        : voxel_size_(voxel_size), max_depth_(max_depth), window_("3D Occupancy Grid") {
    }

    void update(const cv::Mat& depth_map, const cv::Vec3f& position, const cv::Matx33f& rotation) {
        for (int u = 0; u < depth_map.rows; ++u) {
            for (int v = 0; v < depth_map.cols; ++v) {
                float depth = depth_map.at<float>(u, v);
                if (depth <= 0 || depth > max_depth_) continue;

                cv::Vec3f local_point = pixelToPoint(u, v, depth);
                cv::Vec3f world_point = transformPoint(local_point, position, rotation);

                VoxelKey voxel = worldToVoxel(world_point);
                occupancy_grid_[voxel] += 1.0f;
                raycast(position, world_point);
            }
        }
    }

    void visualize() {
        window_.showWidget("Coordinate System", cv::viz::WCoordinateSystem(0.2));

        for (const auto& [key, value] : occupancy_grid_) {
            float occupancy = std::min(1.0f, value / 10.0f);
            cv::Vec3f color(255 * (1 - occupancy), 255 * occupancy, 0);

            cv::Vec3f voxel_center = voxelToWorld(key);
            cv::viz::WCube voxel(cv::Point3f(voxel_center - cv::Vec3f(voxel_size_ / 2)),
                cv::Point3f(voxel_center + cv::Vec3f(voxel_size_ / 2)), true, cv::viz::Color(color[0], color[1], color[2]));
            window_.showWidget("Voxel_" + std::to_string(key.x) + "_" + std::to_string(key.y) + "_" + std::to_string(key.z), voxel);
        }
        window_.spinOnce(1, true);
    }

private:
    float voxel_size_;
    float max_depth_;
    std::unordered_map<VoxelKey, float, VoxelHash> occupancy_grid_;
    cv::viz::Viz3d window_;

    cv::Vec3f pixelToPoint(int u, int v, float depth) {
        float fx = 500.0f, fy = 500.0f;
        float cx = 320.0f, cy = 240.0f;
        float x = (v - cx) * depth / fx;
        float y = (u - cy) * depth / fy;
        return cv::Vec3f(x, y, depth);
    }

    cv::Vec3f transformPoint(const cv::Vec3f& point, const cv::Vec3f& position, const cv::Matx33f& rotation) {
        return rotation * point + position;
    }

    VoxelKey worldToVoxel(const cv::Vec3f& point) {
        return { static_cast<int>(std::round(point[0] / voxel_size_)),
                 static_cast<int>(std::round(point[1] / voxel_size_)),
                 static_cast<int>(std::round(point[2] / voxel_size_)) };
    }

    cv::Vec3f voxelToWorld(const VoxelKey& key) {
        return cv::Vec3f(key.x * voxel_size_, key.y * voxel_size_, key.z * voxel_size_);
    }

    void raycast(const cv::Vec3f& start, const cv::Vec3f& end) {
        cv::Vec3f direction = end - start;
        float distance = cv::norm(direction);
        if (distance == 0) return;

        direction /= distance;
        float step_size = voxel_size_ / 2.0f;

        for (float d = 0; d < distance; d += step_size) {
            cv::Vec3f point = start + direction * d;
            VoxelKey voxel = worldToVoxel(point);
            occupancy_grid_[voxel] -= 0.5f;
        }
    }
};

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
                DisparityMap.convertTo(DisparityMap, CV_64F);
                DisparityMap *= 2.0;
                DisparityMap.convertTo(DisparityMap, T);
            }
        }

        cv::Mat DepthMap;
        {
            int T = DisparityMap.type();
            DisparityMap.convertTo(DisparityMap, CV_64F);
            DepthMap = (double)InputData.Baseline * (double)K.at<float>(0, 0) / DisparityMap;
            DepthMap.convertTo(DepthMap, CV_64F);
            DisparityMap.convertTo(DisparityMap, T);
        }

        static OccupancyGrid3D map;
        DepthMap.convertTo(DepthMap, CV_32F);

        map.update(DepthMap, cv::Vec3f(CamPos[0], -CamPos[1], CamPos[2]), InputData.Rot);
        map.visualize();

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