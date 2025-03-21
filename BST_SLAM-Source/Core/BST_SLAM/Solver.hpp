#pragma once
#include <opencv2/opencv.hpp>

namespace BST_SLAM {
	class Solver {
	public:
		cv::Vec3f SolveISO3(cv::Mat L1, cv::Mat R1, cv::Mat Rot1, cv::Mat L2, cv::Mat R2, cv::Mat Rot2, cv::Mat K, float Baseline, float MaxTrans) {
			cv::Vec3f Trans = cv::Vec3f();

			cv::Size sz = L1.size();
			cv::resize(L1, L1, cv::Size(Resolution, Resolution), 0, 0, cv::INTER_AREA);
			cv::resize(R1, R1, cv::Size(Resolution, Resolution), 0, 0, cv::INTER_AREA);
			cv::resize(L2, L2, cv::Size(Resolution, Resolution), 0, 0, cv::INTER_AREA);
			cv::resize(R2, R2, cv::Size(Resolution, Resolution), 0, 0, cv::INTER_AREA);
			K = K.clone();
			K.convertTo(K, CV_32F);
			float fx = K.at<float>(0, 0);
			float fy = K.at<float>(1, 1);
			K = cv::Mat::eye(3, 3, CV_32F);
			K.at<float>(0, 0) = fx / sz.width * Resolution;
			K.at<float>(1, 1) = fy / sz.height * Resolution;
			K.at<float>(0, 2) = Resolution / 2.0f;
			K.at<float>(1, 2) = Resolution / 2.0f;

			cv::Mat ISO2 = K * (RotDir > 0 ? (Rot2.inv() * Rot1).inv() : ((Rot2.inv() * Rot1).inv()).inv()) * K.inv();
			L1 = L1.clone();
			R1 = R1.clone();
			L2 = L2.clone();
			R2 = R2.clone();
			cv::warpPerspective(L1, L1, ISO2, L1.size());
			cv::warpPerspective(R1, R1, ISO2, R1.size());
			ISO2 = SolveISO2(L1, L2);

			if (!ISO2.empty()) {
				if (SolveISO3(L1, R1, ISO2, K, Baseline, Trans)) {
					if (cv::norm(Trans) > MaxTrans) Trans = cv::Vec3f();
					Trans = (cv::Vec3f)(cv::Mat)(Rot1 * -Trans);
				}
				else Trans = cv::Vec3f();
			}

			Trans = cv::Vec3f(Trans[0], -Trans[1], Trans[2]);
			return Trans;
		}

	private:
		bool SolveISO3(cv::Mat L1, cv::Mat R1, cv::Mat ISO2, cv::Mat K, float Baseline, cv::Vec3f& T) {
			try {
				if (ISO2.rows != 3 || ISO2.cols != 3) return false;

				auto P1 = SolveOSI3(L1, R1, K, Baseline);
				if (P1.size() < 15) return false;

				cv::Vec3f T1, T2, R1, R2;

				auto _P2 = SolveOSI2(P1, cv::Mat::eye(3, 3, CV_32F), K);
				if (_P2.size() < 15) return false;

				cv::solvePnP(P1, _P2, K, cv::Mat(), R1, T1, true, cv::SOLVEPNP_ITERATIVE);

				auto P2 = SolveOSI2(P1, ISO2, K);
				if (P2.size() < 15) return false;

				cv::solvePnP(P1, P2, K, cv::Mat(), R2, T2, true, cv::SOLVEPNP_ITERATIVE);

				T = T2 - T1;
				return true;
			}
			catch (...) { return false; }
		}

		cv::Mat SolveISO2(cv::Mat L1, cv::Mat L2) {
			std::vector<cv::Point2f> _P1, _P2;
			Matches1(L1, L2, _P1, _P2);
			if (_P1.size() < 15 || _P2.size() < 15) return cv::Mat();

			std::vector<cv::Point2f> P1, P2;
			Matches2(_P1, _P2, P1, P2);
			if (P1.size() < 15 || P2.size() < 15) return cv::Mat();

			cv::Mat ISO2 = cv::estimateAffinePartial2D(P1, P2);
			if (ISO2.empty()) return cv::Mat();
			ISO2.convertTo(ISO2, CV_32F);
			cv::vconcat(ISO2, cv::Vec3f(0, 0, 1).t(), ISO2);
			return ISO2;
		}

		int RotDir = -1;
		float ModelSeed = 35;
		float ModelGen = ((std::ceilf(ModelSeed / 10.0f) * 10.0f) - ModelSeed) * 10.0f;
		float Resolution = 500.0f * (1.0f - ModelGen / 100.0f);
		cv::Ptr<cv::ORB> ORB = cv::ORB::create(Resolution * 2.0f, 2.0f - ModelSeed / 100.0f);
		cv::Ptr<cv::BFMatcher> BFMatcher = cv::BFMatcher::create(cv::NORM_HAMMING);

		std::vector<cv::Point3f> SolveOSI3(cv::Mat L1, const cv::Mat& R1, cv::Mat K, float Baseline) {
			std::vector<cv::Point3f> OSI3;

			std::vector<cv::Point2f> _P1, _P2;
			Matches1(L1, R1, _P1, _P2);
			if (_P1.size() < 15) return {};

			std::vector<cv::Point2f> P1, P2;
			Matches2(_P1, _P2, P1, P2);
			if (P1.size() < 15) return {};

			cv::Mat L = cv::Mat::eye(3, 4, CV_32F);
			L.at<float>(0, 3) = Baseline / 2.0f;
			L = K * L;

			cv::Mat R = cv::Mat::eye(3, 4, CV_32F);
			R.at<float>(0, 3) = Baseline / -2.0f;
			R = K * R;

			cv::Mat OSI4;
			cv::triangulatePoints(L, R, P1, P2, OSI4);
			if (OSI4.total() < 15) return {};

			cv::convertPointsFromHomogeneous(OSI4.t(), OSI3);
			return OSI3;
		}

		std::vector<cv::Point2f> SolveOSI2(std::vector<cv::Point3f> OSI3, cv::Mat H, cv::Mat K) {
			std::vector<cv::Point2f> OSI2;
			OSI3 = std::vector<cv::Point3f>(OSI3);
			for (auto& P : OSI3) P.z *= 2.0f;
			cv::projectPoints(OSI3, cv::Vec3f(), cv::Vec3f(), K, cv::noArray(), OSI2);
			if (!H.empty()) cv::perspectiveTransform(OSI2, OSI2, H);
			return OSI2;
		}

		std::vector<cv::DMatch> Matches1(cv::Mat Img1, cv::Mat Img2, std::vector<cv::Point2f>& P1, std::vector<cv::Point2f>& P2) {
			std::vector<cv::KeyPoint> K1, K2;
			cv::Mat D1, D2;
			ORB->detectAndCompute(Img1, cv::noArray(), K1, D1);
			ORB->detectAndCompute(Img2, cv::noArray(), K2, D2);

			if (K1.size() < 15 || K2.size() < 15) return {};

			std::vector<cv::DMatch> _Matches;
			BFMatcher->match(D1, D2, _Matches);
			if (_Matches.size() < 15) return {};

			for (const auto& M : _Matches) {
				P1.push_back(K1[M.queryIdx].pt);
				P2.push_back(K2[M.trainIdx].pt);
			}
			return _Matches;
		}

		void Matches2(std::vector<cv::Point2f> P1, std::vector<cv::Point2f> P2, std::vector<cv::Point2f>& In1, std::vector<cv::Point2f>& In2) {
			if (P1.size() < 15) return;

			cv::Mat In;
			if (cv::findHomography(P1, P2, In, cv::USAC_MAGSAC).empty()) return;

			for (int i = 0; i < (int)In.total(); i++) {
				if (In.at<uchar>(i)) {
					In1.push_back(P1[i]);
					In2.push_back(P2[i]);
				}
			}
		}
	};
}