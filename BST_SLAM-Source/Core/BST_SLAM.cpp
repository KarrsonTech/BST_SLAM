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
    float x, y, z;
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelHash {
    std::size_t operator()(const VoxelKey& key) const {
        return std::hash<float>()(key.x) ^ std::hash<float>()(key.y) ^ std::hash<float>()(key.z);
    }
};

class OccupancyGrid3D {
public:
    OccupancyGrid3D(float voxel_size = 0.05)
        : voxel_size_(voxel_size), window_("3D Occupancy Grid") {
    }

    void update(std::vector<cv::Point3f> Pts3D, const cv::Vec3f& position, const cv::Matx33f& rotation) {
        std::vector<cv::Point3f> occupancy_grid;
        for (const auto& Pt3D : Pts3D)
        {
            cv::Vec3f world_point = transformPoint(Pt3D, position, rotation);
            VoxelKey voxel = worldToVoxel(world_point);
            occupancy_grid.push_back(cv::Point3f(voxelToWorld(voxel)[0], voxelToWorld(voxel)[1], -voxelToWorld(voxel)[2]));
        }
        
        occupancy_grid_.push_back(cv::viz::WCloud(occupancy_grid, cv::viz::Color::green()));
        while (occupancy_grid_.size() >= MapMax)
        {
            occupancy_grid_.erase(occupancy_grid_.begin());
            occupancy_grid_c++;
        }
    }

void visualize() {
    window_.showWidget("Coordinate System", cv::viz::WCoordinateSystem(0.2));

    if (!occupancy_grid_.empty()) {
        int idx = occupancy_grid_c;
        for (const auto& pt : occupancy_grid_)
        {
            window_.showWidget(std::to_string(idx), pt);
            idx++;
        }
    }

    window_.spinOnce(1, true);
}


private:
    float voxel_size_;
    std::vector<cv::viz::WCloud> occupancy_grid_;
    cv::viz::Viz3d window_;
    int occupancy_grid_c = 0;

    cv::Vec3f transformPoint(cv::Vec3f point, cv::Vec3f position, const cv::Matx33f& rotation) {
        point[2] = std::fabs(point[2]);
        point = rotation * point;
        point = position + point;
        return point;
    }

    VoxelKey worldToVoxel(const cv::Vec3f& point) {
        return voxel_size_ > 0 ? VoxelKey
        { 
            std::round(point[0] / voxel_size_),
            std::round(point[1] / voxel_size_),
            std::round(point[2] / voxel_size_) 
        } : VoxelKey
        { 
            point[0],
            point[1],
            point[2] 
        };
    }

    cv::Vec3f voxelToWorld(const VoxelKey& key) {
        return voxel_size_ > 0 ?
            cv::Vec3f(key.x * voxel_size_, key.y * voxel_size_, key.z * voxel_size_) :
            cv::Vec3f(key.x, key.y, key.z);
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

        static OccupancyGrid3D map;

        if (!Solver->Pts3D.empty()) map.update(Solver->Pts3D, CamPos, cv::Quatf(CamRot[3], CamRot[0], CamRot[1], CamRot[2]).toRotMat3x3());
        map.visualize();

        InputData.Left.copyTo(PrevLeft);
        InputData.Right.copyTo(PrevRight);
        InputData.Rot.copyTo(PrevRot);
    }

    return 0;
}