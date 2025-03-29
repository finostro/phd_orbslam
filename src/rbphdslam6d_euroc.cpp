/*
 * Software License Agreement (New BSD License)
 *
 * Copyright (c) 2013, Keith Leung, Felipe Inostroza
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Advanced Mining Technology Center (AMTC), the
 *       Universidad de Chile, nor the names of its contributors may be
 *       used to endorse or promote products derived from this software without
 *       specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE AMTC, UNIVERSIDAD DE CHILE, OR THE COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
 * THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "GaussianGenerators.hpp"
#include "ORB.hpp"
#include "ProcessModel_Odometry6D.hpp"
#include "RBPHDFilter.hpp"
#include "external/argparse.hpp"
#include "measurement_models/MeasurementModel_3D_stereo_orb.hpp"
#include <boost/graph/visitors.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <filesystem>
#include <gtsam/geometry/StereoCamera.h>
#include <memory>
#include <rclcpp/node_options.hpp>
#include <sensor_msgs/msg/detail/point_cloud2__struct.hpp>
#include <stdio.h>
#include <string>
#include <sys/ioctl.h>
#include <visualization_msgs/msg/detail/marker_array__struct.hpp>
#include <yaml-cpp/yaml.h>
#include "misc/EigenYamlSerialization.hpp"

#include<opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <external/Converter.h>
#include <external/ORBextractor.h>

#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <rclcpp/rclcpp.hpp>

#ifdef _PERFTOOLS_CPU
#include <gperftools/profiler.h>
#endif
#ifdef _PERFTOOLS_HEAP
#include <gperftools/heap-profiler.h>
#endif

using namespace rfs;

static const int orb_th_low = 50;
static const int orb_th_high = 100;

// Computes the Hamming distance between two ORB descriptors
static int descriptorDistance(const cv::Mat &a, const cv::Mat &b) {
  const int *pa = a.ptr<int32_t>();
  const int *pb = b.ptr<int32_t>();

  int dist = 0;

  for (int i = 0; i < 8; i++, pa++, pb++) {
    unsigned int v = *pa ^ *pb;
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
  }

  return dist;
}
inline double dist2loglikelihood(int d) {
  double dd = d / 255.0;
  static double a = std::log(1.0 / 8.0);
  static double b = std::log(7.0 / 8.0);
  static double c = std::log(1.0 / 2.0);
  double ret = 14.0 + 20.0 * (dd * a + (1 - dd) * b + c);
  return ret;
}

/**
 * \class Simulator_RBPHDSLAM_6d
 * \brief A 6d SLAM Simulator using the RB-PHD Filter
 * \author Felipe Inostroza
 */
class Simulator_RBPHDSLAM_6d {

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Simulator_RBPHDSLAM_6d() { pFilter_ = NULL; }

  ~Simulator_RBPHDSLAM_6d() {

    if (pFilter_ != NULL) {
      delete pFilter_;
    }
  }

  /** Read the simulator configuration file */
  bool readConfigFile(const char *fileName) {

    cfgFileName_ = fileName;
    YAML::Node node = YAML::LoadFile(fileName);

    logResultsToFile_ = node["config"]["logging"]["logResultsToFile"].as<bool>();
    logTimingToFile_ = node["config"]["logging"]["logTimingToFile"].as<bool>();
    logDirPrefix_ = node["config"]["logging"]["logDirPrefix"].as<std::string>();
    if (*logDirPrefix_.rbegin() != '/')
      logDirPrefix_ += '/';

    // use_gui_ = node["config"]["use_gui"].as<bool>();
    kMax_ = node["config"]["timesteps"].as<int>();
    // dT_ = node["config"]["sec_per_timestep"].as<double>();
    // dTimeStamp_ = TimeStamp(dT_);

    nSegments_ = node["config"]["trajectory"]["nSegments"].as<int>();
    max_dx_ = node["config"]["trajectory"]["max_dx_per_sec"].as<double>();
    max_dy_ = node["config"]["trajectory"]["max_dy_per_sec"].as<double>();
    max_dz_ = node["config"]["trajectory"]["max_dz_per_sec"].as<double>();
    max_dqx_ = node["config"]["trajectory"]["max_dqx_per_sec"].as<double>();
    max_dqy_ = node["config"]["trajectory"]["max_dqy_per_sec"].as<double>();
    max_dqz_ = node["config"]["trajectory"]["max_dqz_per_sec"].as<double>();
    max_dqw_ = node["config"]["trajectory"]["max_dqw_per_sec"].as<double>();
    min_dx_ = node["config"]["trajectory"]["min_dx_per_sec"].as<double>();
    vardx_ = node["config"]["trajectory"]["vardx"].as<double>();
    vardy_ = node["config"]["trajectory"]["vardy"].as<double>();
    vardz_ = node["config"]["trajectory"]["vardz"].as<double>();
    vardqx_ = node["config"]["trajectory"]["vardqx"].as<double>();
    vardqy_ = node["config"]["trajectory"]["vardqy"].as<double>();
    vardqz_ = node["config"]["trajectory"]["vardqz"].as<double>();
    vardqw_ = node["config"]["trajectory"]["vardqw"].as<double>();

    nLandmarks_ = node["config"]["landmarks"]["nLandmarks"].as<int>();
    varlmx_ = node["config"]["landmarks"]["varlmx"].as<double>();
    varlmy_ = node["config"]["landmarks"]["varlmy"].as<double>();
    varlmz_ = node["config"]["landmarks"]["varlmz"].as<double>();

    rangeLimitMax_ = node["config"]["measurements"]["rangeLimitMax"].as<double>();
    rangeLimitMin_ = node["config"]["measurements"]["rangeLimitMin"].as<double>();
    rangeLimitBuffer_ = node["config"]["measurements"]["rangeLimitBuffer"].as<double>();
    Pd_ = node["config"]["measurements"]["probDetection"].as<double>();
    c_ = node["config"]["measurements"]["clutterIntensity"].as<double>();
    varzx_ = node["config"]["measurements"]["varzx"].as<double>();
    varzy_ = node["config"]["measurements"]["varzy"].as<double>();
    varzz_ = node["config"]["measurements"]["varzz"].as<double>();
    
    nParticles_ = node["config"]["filter"]["nParticles"].as<int>();

    pNoiseInflation_ = node["config"]["filter"]["predict"]["processNoiseInflationFactor"].as<double>();
    birthGaussianWeight_ = node["config"]["filter"]["predict"]["birthGaussianWeight"].as<double>();

    zNoiseInflation_ = node["config"]["filter"]["update"]["measurementNoiseInflationFactor"].as<double>();
    innovationRangeThreshold_ = node["config"]["filter"]["update"]["KalmanFilter"]["innovationThreshold"]["range"].as<double>();
    innovationBearingThreshold_ = node["config"]["filter"]["update"]["KalmanFilter"]["innovationThreshold"]["bearing"].as<double>();
    newGaussianCreateInnovMDThreshold_ = node["config"]["filter"]["update"]["GaussianCreateInnovMDThreshold"].as<double>();

    importanceWeightingEvalPointCount_ = node["config"]["filter"]["weighting"]["nEvalPt"].as<int>();
    importanceWeightingEvalPointGuassianWeight_ = node["config"]["filter"]["weighting"]["minWeight"].as<double>();
    importanceWeightingMeasurementLikelihoodMDThreshold_ = node["config"]["filter"]["weighting"]["threshold"].as<double>();
    useClusterProcess_ = node["config"]["filter"]["weighting"]["useClusterProcess"].as<bool>();
    
    effNParticleThreshold_ = node["config"]["filter"]["resampling"]["effNParticle"].as<double>();
    minUpdatesBeforeResample_ = node["config"]["filter"]["resampling"]["minTimesteps"].as<int>();

    gaussianMergingThreshold_ = node["config"]["filter"]["merge"]["threshold"].as<double>();
    gaussianMergingCovarianceInflationFactor_ = node["config"]["filter"]["merge"]["covInflationFactor"].as<double>();

    gaussianPruningThreshold_ = node["config"]["filter"]["prune"]["threshold"].as<double>();

    eurocFolder_ = node["config"]["euroc"]["folder"].as<std::string>();
    eurocTimestampsFilename_ = node["config"]["euroc"]["timestampsFilename"].as<std::string>();
    
    
	for (auto camera : node["camera_params"]) {
		CameraParams params;
		params.fx = camera["fx"].as<double>();
		params.fy = camera["fy"].as<double>();
		params.cx = camera["cx"].as<double>();
		params.cy = camera["cy"].as<double>();
		params.k1 = camera["k1"].as<double>();
		params.k2 = camera["k2"].as<double>();
		params.p1 = camera["p1"].as<double>();
		params.p2 = camera["p2"].as<double>();
		params.originalImSize.width = camera["width"].as<int>();
		params.originalImSize.height = camera["height"].as<int>();

		params.newImSize = params.originalImSize;

		if (!YAML::convert<Eigen::MatrixXd>::decode(camera["cv_c0_to_camera"],
				params.cv_c0_to_camera_eigen)) {
			std::cerr << "could not load principal_point \n";
			exit(1);
		}
		cv::eigen2cv(params.cv_c0_to_camera_eigen, params.cv_c0_to_camera);

		cv::Mat dist_coeffs(4, 1, CV_64F);
		dist_coeffs.at<float>(0, 0) = params.k1;
		dist_coeffs.at<float>(1, 0) = params.k2;
		dist_coeffs.at<float>(2, 0) = params.p1;
		dist_coeffs.at<float>(3, 0) = params.p2;
		params.opencv_distort_coeffs =
				(cv::Mat_<double>(4, 1) << params.k1, params.k2, params.p1, params.p2);

		params.opencv_calibration =
				(cv::Mat_<double>(3, 3) << (float) params.fx, 0.f, (float) params.cx, 0.f, (float) params.fy, (float) params.cy, 0.f, 0.f, 1.f);

		camera_parameters_.push_back(params);
	}

	cv::Mat cvTlr = camera_parameters_[1].cv_c0_to_camera;
	Sophus::SE3d Tlr = ORB_SLAM3::Converter::toSophusd(cvTlr);

	cv::Mat R12 = cvTlr.rowRange(0, 3).colRange(0, 3);
	R12.convertTo(R12, CV_64F);
	cv::Mat t12 = cvTlr.rowRange(0, 3).col(3);
	t12.convertTo(t12, CV_64F);

	stereo_baseline_ = Tlr.translation().norm();
	stereo_baseline_f_ = stereo_baseline_
			* camera_parameters_[0].fx;

	Eigen::Vector2d principal_point = { camera_parameters_[0].cx,
			camera_parameters_[0].cy };

	cv::Mat R_r1_u1, R_r2_u2;
	cv::Mat P1, P2, Q;

	cv::stereoRectify(camera_parameters_[0].opencv_calibration,
			camera_parameters_[0].opencv_distort_coeffs,
			camera_parameters_[1].opencv_calibration,
			camera_parameters_[1].opencv_distort_coeffs,
			camera_parameters_[0].newImSize, R12, t12, R_r1_u1, R_r2_u2,
			P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1,
			camera_parameters_[0].newImSize);

	cv::initUndistortRectifyMap(camera_parameters_[0].opencv_calibration,
			camera_parameters_[0].opencv_distort_coeffs, R_r1_u1,
			P1.rowRange(0, 3).colRange(0, 3),
			camera_parameters_[0].newImSize, CV_32F,
			camera_parameters_[0].M1, camera_parameters_[0].M2);
	cv::initUndistortRectifyMap(camera_parameters_[1].opencv_calibration,
			camera_parameters_[1].opencv_distort_coeffs, R_r2_u2,
			P2.rowRange(0, 3).colRange(0, 3),
			camera_parameters_[1].newImSize, CV_32F,
			camera_parameters_[1].M1, camera_parameters_[1].M2);

	std::cout << "cam0 opencv_calibration: "
			<< camera_parameters_[0].opencv_calibration << "\n";
	std::cout << "cam1 opencv_calibration: "
			<< camera_parameters_[1].opencv_calibration << "\n";
	std::cout << "cam0 opencv_distort_coeffs: "
			<< camera_parameters_[0].opencv_distort_coeffs << "\n";
	std::cout << "cam1 opencv_distort_coeffs: "
			<< camera_parameters_[1].opencv_distort_coeffs << "\n";

//    std::cout << "cam0 M1: " << camera_parameters_[0].M1 << "\n";
//    std::cout << "cam0 M2: " << camera_parameters_[0].M2 << "\n";
//    std::cout << "cam1 M1: " << camera_parameters_[1].M1 << "\n";
//    std::cout << "cam1 M2: " << camera_parameters_[1].M2 << "\n";

	orb_extractor.nFeatures = node["ORBextractor.nFeatures"].as<int>();
	orb_extractor.scaleFactor = node["ORBextractor.scaleFactor"].as<
			double>();
	orb_extractor.nLevels = node["ORBextractor.nLevels"].as<int>();
	orb_extractor.iniThFAST = node["ORBextractor.iniThFAST"].as<int>();
	orb_extractor.minThFAST = node["ORBextractor.minThFAST"].as<int>();
	stereo_init_max_depth_ = node["stereo_init_max_depth"].as<double>();

	mpORBextractorLeft = new ORB_SLAM3::ORBextractor(
			orb_extractor.nFeatures, orb_extractor.scaleFactor,
			orb_extractor.nLevels, orb_extractor.iniThFAST,
			orb_extractor.minThFAST);
	mpORBextractorRight = new ORB_SLAM3::ORBextractor(
			orb_extractor.nFeatures, orb_extractor.scaleFactor,
			orb_extractor.nLevels, orb_extractor.iniThFAST,
			orb_extractor.minThFAST);

	orbExtractorInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
	orbExtractorScaleFactors = mpORBextractorLeft->GetScaleFactors();


    return true;
  }

  /** Generate a 6d trajectory in 3d space
   *  \param[in] randSeed random seed for generating trajectory
   */
  // void generateTrajectory(int randSeed = 0) {
  //
  // 	srand48(randSeed);
  // 	initializeGaussianGenerators();
  //
  // 	TimeStamp t;
  // 	int seg = 0;
  // 	MotionModel_Odometry6d::TState::Mat Q;
  // 	Q.setZero();
  // 	Q(0, 0) = vardx_;
  // 	Q(1, 1) = vardy_;
  // 	Q(2, 2) = vardz_;
  // 	Q(3, 3) = vardqw_;
  // 	Q(4, 4) = vardqx_;
  // 	Q(5, 5) = vardqy_;
  // 	Q(6, 6) = vardqz_;
  // 	MotionModel_Odometry6d motionModel(Q);
  // 	MotionModel_Odometry6d::TInput input_k(t);
  // 	MotionModel_Odometry6d::TState pose_k(t);
  // 	pose_k[6]=1.0;
  // 	MotionModel_Odometry6d::TState pose_km(t);
  // 	pose_km[6]=1.0;
  // 	groundtruth_displacement_.reserve(kMax_);
  // 	groundtruth_pose_.reserve(kMax_);
  // 	groundtruth_displacement_.push_back(input_k);
  // 	groundtruth_pose_.push_back(pose_k);
  //
  // 	for (int k = 1; k < kMax_; k++) {
  //
  // 		t += dTimeStamp_;
  //
  // 		if (k <= 50) {
  // 			double dx = 0;
  // 			double dy = 0;
  // 			double dz = 0;
  // 			double dqx = 0;
  // 			double dqy = 0;
  // 			double dqz = 0;
  // 			double dqw = 1;
  // 			MotionModel_Odometry6d::TInput::Vec d;
  // 			MotionModel_Odometry6d::TInput::Vec dCovDiag;
  // 			d << dx, dy, dz, dqx, dqy, dqz, dqw;
  // 			dCovDiag << 0, 0, 0, 0, 0, 0, 0;
  // 			input_k = MotionModel_Odometry6d::TInput(d,
  // 					dCovDiag.asDiagonal(), k);
  // 		} else if (k >= kMax_ / nSegments_ * seg) {
  // 			seg++;
  // 			double dx = drand48() * (max_dx_ - min_dx_) * dT_
  // 					+ min_dx_ * dT_;
  //
  // 			double dy = (drand48() * max_dy_ * 2 - max_dy_) * dT_;
  // 			double dz = (drand48() * max_dz_ * 2 - max_dz_) * dT_;
  // 			double dqx = (drand48() * max_dqx_ * 2 - max_dqx_) * dT_;
  // 			double dqy = (drand48() * max_dqy_ * 2 - max_dqy_) * dT_;
  // 			double dqz = (drand48() * max_dqz_ * 2 - max_dqz_) * dT_;
  // 			double dqw = 1+(drand48() * max_dqw_ * 2 - max_dqw_) * dT_;
  //
  // 			MotionModel_Odometry6d::TInput::Vec d;
  // 			MotionModel_Odometry6d::TInput::Vec dCovDiag;
  // 			d << dx, dy, dz, dqx, dqy, dqz, dqw;
  //
  // 			dCovDiag << Q(0, 0), Q(1, 1), Q(2, 2), Q(3, 3), Q(4, 4), Q(5,
  // 					5), Q(6, 6);
  // 			input_k = MotionModel_Odometry6d::TInput(d,
  // 					dCovDiag.asDiagonal(), k);
  // 		}
  //
  // 		groundtruth_displacement_.push_back(input_k);
  // 		groundtruth_displacement_.back().setTime(t);
  //
  // 		MotionModel_Odometry6d::TState x_k;
  // 		motionModel.step(x_k, groundtruth_pose_[k - 1], input_k,
  // 				dTimeStamp_);
  // 		groundtruth_pose_.push_back(x_k);
  // 		groundtruth_pose_.back().setTime(t);
  //
  // 	}
  //
  // }

  /** Generate odometry measurements */
  // void generateOdometry() {
  //
  // 	odometry_.reserve(kMax_);
  // 	MotionModel_Odometry6d::TInput zero;
  // 	MotionModel_Odometry6d::TInput::Vec u0;
  // 	u0.setZero();
  // 	zero.set(u0, 0);
  // 	odometry_.push_back(zero);
  //
  // 	MotionModel_Odometry6d::TState::Mat Q;
  // 	Q.setZero();
  // 	Q(0, 0) = vardx_;
  // 	Q(1, 1) = vardy_;
  // 	Q(2, 2) = vardz_;
  // 	Q(3, 3) = vardqw_;
  // 	Q(4, 4) = vardqx_;
  // 	Q(5, 5) = vardqy_;
  // 	Q(6, 6) = vardqz_;
  // 	MotionModel_Odometry6d motionModel(Q);
  // 	deadReckoning_pose_.reserve(kMax_);
  // 	deadReckoning_pose_.push_back(groundtruth_pose_[0]);
  //
  // 	TimeStamp t;
  //
  // 	for (int k = 1; k < kMax_; k++) {
  //
  // 		t += dTimeStamp_;
  // 		double dt = dTimeStamp_.getTimeAsDouble();
  //
  // 		MotionModel_Odometry6d::TInput in = groundtruth_displacement_[k];
  // 		MotionModel_Odometry6d::TState::Mat Qk = Q * dt * dt;
  // 		in.setCov(Qk);
  // 		MotionModel_Odometry6d::TInput out;
  // 		in.sample(out);
  //
  // 		odometry_.push_back(out);
  //
  // 		MotionModel_Odometry6d::TState p;
  // 		motionModel.step(p, deadReckoning_pose_[k - 1], odometry_[k],
  // 				dTimeStamp_);
  // 		p.setTime(t);
  // 		deadReckoning_pose_.push_back(p);
  // 	}
  //
  // }

  /** Generate landmarks */
  // void generateLandmarks() {
  //
  // 	MeasurementModel_3D_stereo_orb measurementModel(varzx_, varzy_, varzz_);
  // 	MeasurementModel_3D_stereo_orb::TPose pose;
  //
  // 	groundtruth_landmark_.reserve(nLandmarks_);
  //
  // 	int nLandmarksCreated = 0;
  // 	for (int k = 1; k < kMax_; k++) {
  //
  // 		if (k >= kMax_ / nLandmarks_ * nLandmarksCreated) {
  //
  // 			MeasurementModel_3D_stereo_orb::TPose pose;
  // 			MeasurementModel_3D_stereo_orb::TMeasurement measurementToCreateLandmark;
  // 			MeasurementModel_3D_stereo_orb::TMeasurement::Vec z;
  // 			do {
  // 				for (int i = 0; i < z.size(); i++)
  // 					z(i) = drand48() * 2 * rangeLimitMax_ - rangeLimitMax_;
  // 			} while (z.norm() > rangeLimitMax_);
  //
  // 			measurementToCreateLandmark.set(z);
  // 			MeasurementModel_3D_stereo_orb::TLandmark lm;
  //
  // 			measurementModel.inverseMeasure(groundtruth_pose_[k],
  // 					measurementToCreateLandmark, lm);
  //
  // 			groundtruth_landmark_.push_back(lm);
  //
  // 			nLandmarksCreated++;
  //
  // 		}
  //
  // 	}
  //
  // }

  /** Generate landmark measurements */
  // void generateMeasurements() {
  //
  // 	MeasurementModel_3D_stereo_orb measurementModel(varzx_, varzy_, varzz_);
  // 	MeasurementModel_3D_stereo_orb::TMeasurement::Mat R;
  // 	measurementModel.getNoise(R);
  // 	measurementModel.config.rangeLimMax_ = rangeLimitMax_;
  // 	measurementModel.config.rangeLimMin_ = rangeLimitMin_;
  // 	measurementModel.config.probabilityOfDetection_ = Pd_;
  // 	measurementModel.config.uniformClutterIntensity_ = c_;
  // 	double meanClutter = measurementModel.clutterIntensityIntegral();
  //
  // 	double expNegMeanClutter = exp(-meanClutter);
  // 	double poissonPmf[100];
  // 	double poissonCmf[100];
  // 	double mean_pow_i = 1;
  // 	double i_factorial = 1;
  // 	poissonPmf[0] = expNegMeanClutter;
  // 	poissonCmf[0] = poissonPmf[0];
  // 	for (int i = 1; i < 100; i++) {
  // 		mean_pow_i *= meanClutter;
  // 		i_factorial *= i;
  // 		poissonPmf[i] = mean_pow_i / i_factorial * expNegMeanClutter;
  // 		poissonCmf[i] = poissonCmf[i - 1] + poissonPmf[i];
  // 	}
  //
  // 	lmkFirstObsTime_.resize(groundtruth_landmark_.size());
  // 	for (int m = 0; m < lmkFirstObsTime_.size(); m++) {
  // 		lmkFirstObsTime_[m] = -1;
  // 	}
  //
  // 	TimeStamp t;
  //
  // 	for (int k = 1; k < kMax_; k++) {
  //
  // 		t += dTimeStamp_;
  //
  // 		groundtruth_pose_[k];
  //
  // 		// Real detections
  // 		for (int m = 0; m < groundtruth_landmark_.size(); m++) {
  //
  // 			bool success;
  // 			MeasurementModel_3D_stereo_orb::TMeasurement z_m_k;
  // 			success = measurementModel.sample(groundtruth_pose_[k],
  // 					groundtruth_landmark_[m], z_m_k);
  // 			if (success) {
  //
  // 				if ( drand48() <= Pd_) {
  // 					z_m_k.setTime(t);
  // 					// z_m_k.setCov(R);
  // 					measurements_.push_back(z_m_k);
  // 				}
  //
  // 				if (lmkFirstObsTime_[m] == -1) {
  // 					lmkFirstObsTime_[m] = t.getTimeAsDouble();
  // 				}
  // 			}
  //
  // 		}
  //
  // 		// False alarms
  // 		double randomNum = drand48();
  // 		int nClutterToGen = 0;
  // 		while (randomNum > poissonCmf[nClutterToGen]) {
  // 			nClutterToGen++;
  // 		}
  // 		for (int i = 0; i < nClutterToGen; i++) {
  //
  // 			MeasurementModel_3D_stereo_orb::TMeasurement z_clutter;
  // 			MeasurementModel_3D_stereo_orb::TMeasurement::Vec z;
  // 			do {
  // 				for (int i = 0; i < z.size(); i++)
  // 					z(i) = drand48() * 2 * rangeLimitMax_ - rangeLimitMax_;
  // 			} while (z.norm() < rangeLimitMax_ && z.norm() > rangeLimitMin_);
  //
  // 			z_clutter.set(z, t);
  // 			measurements_.push_back(z_clutter);
  //
  // 		}
  //
  // 	}
  //
  // }

  /** Data Logging */
  void exportSimData() {

    if (logResultsToFile_ || logTimingToFile_) {
      std::filesystem::path dir(logDirPrefix_);
      std::filesystem::create_directories(dir);
      std::filesystem::path cfgFilePathSrc(cfgFileName_);
      std::string cfgFileDst(logDirPrefix_);
      cfgFileDst += "simSettings.xml";
      std::filesystem::path cfgFilePathDst(cfgFileDst.data());
      std::filesystem::copy_file(cfgFilePathSrc, cfgFilePathDst, std::filesystem::copy_options::overwrite_existing);
    }

    if (!logResultsToFile_)
      return;

    TimeStamp t;
    //
    // FILE* pGTPoseFile;
    // std::string filenameGTPose(logDirPrefix_);
    // filenameGTPose += "gtPose.dat";
    // pGTPoseFile = fopen(filenameGTPose.data(), "w");
    // MotionModel_Odometry6d::TState::Vec x;
    // for (int i = 0; i < groundtruth_pose_.size(); i++) {
    // 	groundtruth_pose_[i].get(x, t);
    // 	fprintf(pGTPoseFile, "%f   %f   %f   %f   %f   %f   %f   %f\n",
    // 			t.getTimeAsDouble(), x(0), x(1), x(2), x(3), x(4), x(5),
    // 			x(6));
    // }
    // fclose(pGTPoseFile);
    //
    // FILE* pGTLandmarkFile;
    // std::string filenameGTLandmark(logDirPrefix_);
    // filenameGTLandmark += "gtLandmark.dat";
    // pGTLandmarkFile = fopen(filenameGTLandmark.data(), "w");
    // MeasurementModel_3D_stereo_orb::TLandmark::Vec m;
    // for (int i = 0; i < groundtruth_landmark_.size(); i++) {
    // 	groundtruth_landmark_[i].get(m);
    // 	fprintf(pGTLandmarkFile, "%f   %f   %f   %f\n", m(0), m(1), m(2),
    // 			lmkFirstObsTime_[i]);
    // }
    // fclose(pGTLandmarkFile);

    FILE *pOdomFile;
    std::string filenameOdom(logDirPrefix_);
    filenameOdom += "odometry.dat";
    pOdomFile = fopen(filenameOdom.data(), "w");
    MotionModel_Odometry6d::TInput::Vec u;
    for (int i = 0; i < odometry_.size(); i++) {
      odometry_[i].get(u, t);
      fprintf(pOdomFile, "%f   %f   %f   %f   %f   %f   %f   %f\n", t.getTimeAsDouble(), u(0), u(1), u(2), u(3), u(4), u(5),
              u(6));
    }
    fclose(pOdomFile);

    FILE *pMeasurementFile;
    std::string filenameMeasurement(logDirPrefix_);
    filenameMeasurement += "measurement.dat";
    pMeasurementFile = fopen(filenameMeasurement.data(), "w");
    MeasurementModel_3D_stereo_orb::TMeasurement::Vec z;
    for (int i = 0; i < measurements_.size(); i++) {
      measurements_[i].get(z, t);
      fprintf(pMeasurementFile, "%f   %f   %f   %f\n", t.getTimeAsDouble(), z(0), z(1), z(2));
    }
    fclose(pMeasurementFile);

    FILE *pDeadReckoningFile;
    std::string filenameDeadReckoning(logDirPrefix_);
    filenameDeadReckoning += "deadReckoning.dat";
    pDeadReckoningFile = fopen(filenameDeadReckoning.data(), "w");
    MotionModel_Odometry6d::TState::Vec odo;
    for (int i = 0; i < deadReckoning_pose_.size(); i++) {
      deadReckoning_pose_[i].get(odo, t);
      fprintf(pDeadReckoningFile, "%f   %f   %f   %f   %f   %f   %f   %f\n", t.getTimeAsDouble(), odo(0), odo(1), odo(2), odo(3),
              odo(4), odo(5), odo(6));
    }
    fclose(pDeadReckoningFile);
  }

  /** RB-PHD Filter Setup */
  void setupRBPHDFilter() {

    pFilter_ = new RBPHDFilter<MotionModel_Odometry6d, StaticProcessModel<Landmark3d>, MeasurementModel_3D_stereo_orb,
                               KalmanFilter<StaticProcessModel<Landmark3d>, MeasurementModel_3D_stereo_orb>>(nParticles_);

    double dt = dTimeStamp_.getTimeAsDouble();

    // configure robot motion model (only need to set once since timesteps are constant)
    MotionModel_Odometry6d::TState::Mat Q;
    Q.setZero();
    Q(0, 0) = vardx_;
    Q(1, 1) = vardy_;
    Q(2, 2) = vardz_;
    Q(3, 3) = vardqx_;
    Q(4, 4) = vardqy_;
    Q(5, 5) = vardqz_;
    Q(6, 6) = vardqw_;
    Q *= (pNoiseInflation_ * dt * dt);
    pFilter_->getProcessModel()->setNoise(Q);

    // configure landmark process model (only need to set once since timesteps are constant)
    Landmark3d::Mat Q_lm;
    Q_lm.setZero();
    Q_lm(0, 0) = varlmx_;
    Q_lm(1, 1) = varlmy_;
    Q_lm(2, 2) = varlmz_;
    Q_lm = Q_lm * dt * dt;
    pFilter_->getLmkProcessModel()->setNoise(Q_lm);

    // configure measurement model
    MeasurementModel_3D_stereo_orb::TMeasurement::Mat R;
    R << varzx_, 0, 0, 0, varzy_, 0, 0, 0, varzz_;
    R *= zNoiseInflation_;
    pFilter_->getMeasurementModel()->setNoise(R);
    pFilter_->getMeasurementModel()->config.probabilityOfDetection_ = Pd_;
    pFilter_->getMeasurementModel()->config.uniformClutterIntensity_ = c_;
    pFilter_->getMeasurementModel()->config.rangeLimMax_ = rangeLimitMax_;
    pFilter_->getMeasurementModel()->config.rangeLimMin_ = rangeLimitMin_;
    pFilter_->getMeasurementModel()->config.rangeLimBuffer_ = rangeLimitBuffer_;

    auto camera_params = boost::make_shared<gtsam::Cal3_S2Stereo>(
        camera_parameters_[0].fx, camera_parameters_[0].fy,
        0, // 0 skew
        camera_parameters_[0].cx, camera_parameters_[0].cy,
        stereo_baseline_);

    gtsam::StereoCamera stereocamera(gtsam::Pose3(), camera_params);

    pFilter_->getMeasurementModel()->config.camera.camera = stereocamera;


    // configure the Kalman filter for landmark updates
    /* Thresholds not implemented!!
     pFilter_->getKalmanFilter()->config.rangeInnovationThreshold_ = innovationRangeThreshold_;
     pFilter_->getKalmanFilter()->config.bearingInnovationThreshold_ = innovationBearingThreshold_;
     */
    // configure the filter
    pFilter_->config.birthGaussianWeight_ = birthGaussianWeight_;
    pFilter_->setEffectiveParticleCountThreshold(effNParticleThreshold_);
    pFilter_->config.minUpdatesBeforeResample_ = minUpdatesBeforeResample_;
    pFilter_->config.newGaussianCreateInnovMDThreshold_ = newGaussianCreateInnovMDThreshold_;
    pFilter_->config.importanceWeightingMeasurementLikelihoodMDThreshold_ = importanceWeightingMeasurementLikelihoodMDThreshold_;
    pFilter_->config.importanceWeightingEvalPointCount_ = importanceWeightingEvalPointCount_;
    pFilter_->config.importanceWeightingEvalPointGuassianWeight_ = importanceWeightingEvalPointGuassianWeight_;
    pFilter_->config.gaussianMergingThreshold_ = gaussianMergingThreshold_;
    pFilter_->config.gaussianMergingCovarianceInflationFactor_ = gaussianMergingCovarianceInflationFactor_;
    pFilter_->config.gaussianPruningThreshold_ = gaussianPruningThreshold_;
    pFilter_->config.useClusterProcess_ = useClusterProcess_;

    // Visualization
    if (use_ros_gui_) {
	  marker_pub_ = visualization_node_->create_publisher<visualization_msgs::msg::MarkerArray>("phd_orbslam_minimal/marker", 10);
	  landmark_cloud_pub_ = visualization_node_->create_publisher<sensor_msgs::msg::PointCloud2>("landmark_cloud", 10);
    }

  }

  visualization_msgs::msg::Marker makeFrustumMarker() {

    auto best_particle = pFilter_->getBestParticle();

    auto position = best_particle->getPose()->getPos();
    Eigen::Quaterniond orientation(  best_particle->getPose()->getRot());
    std::cout << "position: " << position.transpose() << "\n";
    std::cout << "orientation: " << orientation.w() << " " << orientation.x() << " " << orientation.y() << " " << orientation.z() << "\n";

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = rclcpp::Time(0);
    marker.ns = "frustum";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = position[0];
    marker.pose.position.y = position[1];
    marker.pose.position.z = position[2];
    marker.pose.orientation.x = orientation.x();
    marker.pose.orientation.y = orientation.y();
    marker.pose.orientation.z = orientation.z();
    marker.pose.orientation.w = orientation.w();  
    
    std::vector<gtsam::Point3> frustum_points_in_camera_frame;
    std::vector<gtsam::Point3> marker_points;


    frustum_points_in_camera_frame.resize(8);
    

    auto measurementModel = pFilter_->getMeasurementModel();
    gtsam::StereoPoint2 stereopoint(-.99, -1.0, -1.);
    // std::cout << "camera.baseline: " << measurementModel->config.camera.camera.baseline() << "\n";
    // std::cout << "camera.calibration: " << measurementModel->config.camera.camera.calibration() << "\n";
    // std::cout << "stereopoint: " << stereopoint << "\n";
    frustum_points_in_camera_frame[0] = measurementModel->config.camera.camera.backproject(stereopoint);
    frustum_points_in_camera_frame[0] =
        frustum_points_in_camera_frame[0] * (measurementModel->config.rangeLimMin_ / frustum_points_in_camera_frame[0].norm());

    stereopoint = gtsam::StereoPoint2(1, 0.99, -1.);
    frustum_points_in_camera_frame[1] = measurementModel->config.camera.camera.backproject(stereopoint);
    frustum_points_in_camera_frame[1] =
        frustum_points_in_camera_frame[1] * (measurementModel->config.rangeLimMin_ / frustum_points_in_camera_frame[1].norm());

    stereopoint = gtsam::StereoPoint2(1, 0.99, 1.0);
    frustum_points_in_camera_frame[2] = measurementModel->config.camera.camera.backproject(stereopoint);
    frustum_points_in_camera_frame[2] =
        frustum_points_in_camera_frame[2] * (measurementModel->config.rangeLimMin_ / frustum_points_in_camera_frame[2].norm());

    stereopoint = gtsam::StereoPoint2(-0.99, -1.0, 1);
    frustum_points_in_camera_frame[3] = measurementModel->config.camera.camera.backproject(stereopoint);
    frustum_points_in_camera_frame[3] =
        frustum_points_in_camera_frame[3] * (measurementModel->config.rangeLimMin_ / frustum_points_in_camera_frame[3].norm());


    frustum_points_in_camera_frame[4] =
        frustum_points_in_camera_frame[0] * (measurementModel->config.rangeLimMax_ / frustum_points_in_camera_frame[0].norm());
    frustum_points_in_camera_frame[5] = frustum_points_in_camera_frame[1] * (measurementModel->config.rangeLimMax_ / frustum_points_in_camera_frame[1].norm());
    frustum_points_in_camera_frame[6] = frustum_points_in_camera_frame[2] * (measurementModel->config.rangeLimMax_ / frustum_points_in_camera_frame[2].norm());
    frustum_points_in_camera_frame[7] = frustum_points_in_camera_frame[3] * (measurementModel->config.rangeLimMax_ / frustum_points_in_camera_frame[3].norm());

    marker_points.push_back(frustum_points_in_camera_frame[0]);
    marker_points.push_back(frustum_points_in_camera_frame[4]);
    
    marker_points.push_back(frustum_points_in_camera_frame[4]);
    marker_points.push_back(frustum_points_in_camera_frame[5]);

    marker_points.push_back(frustum_points_in_camera_frame[0]);
    marker_points.push_back(frustum_points_in_camera_frame[1]);


    marker_points.push_back(frustum_points_in_camera_frame[1]);
    marker_points.push_back(frustum_points_in_camera_frame[5]);

    marker_points.push_back(frustum_points_in_camera_frame[5]);
    marker_points.push_back(frustum_points_in_camera_frame[6]);

    marker_points.push_back(frustum_points_in_camera_frame[1]);
    marker_points.push_back(frustum_points_in_camera_frame[2]);


    marker_points.push_back(frustum_points_in_camera_frame[2]);
    marker_points.push_back(frustum_points_in_camera_frame[3]);

    marker_points.push_back(frustum_points_in_camera_frame[2]);
    marker_points.push_back(frustum_points_in_camera_frame[6]);

    marker_points.push_back(frustum_points_in_camera_frame[6]);
    marker_points.push_back(frustum_points_in_camera_frame[7]);


    marker_points.push_back(frustum_points_in_camera_frame[3]);
    marker_points.push_back(frustum_points_in_camera_frame[0]);

    marker_points.push_back(frustum_points_in_camera_frame[3]);
    marker_points.push_back(frustum_points_in_camera_frame[7]);

    marker_points.push_back(frustum_points_in_camera_frame[7]);
    marker_points.push_back(frustum_points_in_camera_frame[4]);



    marker.points.resize(frustum_points_in_camera_frame.size());


    std::cout <<  "frustum\n";
    for (size_t  i = 0 ; i < frustum_points_in_camera_frame.size(); i++){
      std::cout << frustum_points_in_camera_frame[i].transpose() << "\n";
      marker.points[i].x = frustum_points_in_camera_frame[i][0];
      marker.points[i].y = frustum_points_in_camera_frame[i][1];
      marker.points[i].z = frustum_points_in_camera_frame[i][2];
    }

    marker.scale.x = 0.1;
    marker.scale.y = 0.0;
    marker.scale.z = 0.0;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    return marker;
  }
  std::unique_ptr<sensor_msgs::msg::PointCloud2> makeRosPointcloud(){

    auto best_particle = pFilter_->getBestParticle();
    auto gaussian_mixture = best_particle->getData();
    int num_gaussians = gaussian_mixture->getGaussianCount();
    int num_points = 0;

    for (int i = 0; i < num_gaussians; i++) {
      auto weight = gaussian_mixture->getWeight(i);
      if ( weight == 0 )
	continue;
    
      num_points++;

    }



    
    auto  pointcloud_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    pointcloud_msg->header.frame_id = "map";
    pointcloud_msg->header.stamp = rclcpp::Time(0);
    
    
    sensor_msgs::PointCloud2Modifier modifier(*pointcloud_msg);
    modifier.setPointCloud2Fields(4, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1, sensor_msgs::msg::PointField::FLOAT32, "z", 1, sensor_msgs::msg::PointField::FLOAT32, "intensity", 1, sensor_msgs::msg::PointField::FLOAT32);
    modifier.resize(num_points);


    sensor_msgs::PointCloud2Iterator<float> iter_x(*pointcloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*pointcloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*pointcloud_msg, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_intensity(*pointcloud_msg, "intensity");
    

    int np = 0 ;
    for (int i = 0; i < num_gaussians; i++) {
      auto weight = gaussian_mixture->getWeight(i);
      if ( weight == 0 )
	continue;
    
      auto mean = gaussian_mixture->getGaussian(i)->get();
      iter_x[np] = mean.x();
      iter_y[np] = mean.y();
      iter_z[np] = mean.z();
      iter_intensity[np] = weight;
	
      np++;

    }


    return std::move(pointcloud_msg);
  }

  std::unique_ptr<visualization_msgs::msg::MarkerArray> makeRosMarkerArray(){
    auto marker_array_msg = std::make_unique<visualization_msgs::msg::MarkerArray>();
    marker_array_msg->markers.push_back(makeFrustumMarker());
    
    return std::move(marker_array_msg);
  }

  void stereoMatchesToMeasurments(std::vector<cv::KeyPoint> &keypoints_left, std::vector<cv::KeyPoint> &keypoints_right,
                            cv::Mat &descriptors_left, cv::Mat &descriptors_right,
                            std::vector<cv::DMatch> &matches_left_to_right, std::vector<MeasurementModel_3D_stereo_orb::TMeasurement> &measurements, double time) {

    std::cout << "number of matches: " << matches_left_to_right.size() << "\n";
    for (int i = 0; i < matches_left_to_right.size(); i++) {
      
      MeasurementModel_3D_stereo_orb::TMeasurement measurement;
      MeasurementModel_3D_stereo_orb::TMeasurement::Vec z;
      z << keypoints_left[matches_left_to_right[i].queryIdx].pt.x, keypoints_left[matches_left_to_right[i].queryIdx].pt.y, keypoints_right[matches_left_to_right[i].trainIdx].pt.x;
      measurement.set(z);
      measurement.setTime(time);
      measurements.push_back(measurement);
    }
  }

  void computeStereoMatches(std::vector<cv::KeyPoint> &keypoints_left, std::vector<cv::KeyPoint> &keypoints_right,
                            cv::Mat &descriptors_left, cv::Mat &descriptors_right,
                            std::vector<cv::DMatch> &matches_left_to_right) {

    std::vector<float> uRight;
    std::vector<float> depth;
    // image bounds

    double mnMinX;
    double mnMaxX;
    double mnMinY;
    double mnMaxY;

    // Scale
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    uRight = std::vector<float>(keypoints_left.size(), -1.0f);
    depth = std::vector<float>(keypoints_left.size(), -1.0f);

    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mnScaleLevels = mpORBextractorLeft->GetLevels();

    static constexpr int thOrbDist = (orb_th_high + orb_th_low) / 2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    std::vector<std::vector<std::size_t>> vRowIndices(nRows, std::vector<std::size_t>());

    for (int i = 0; i < nRows; i++)
      vRowIndices[i].reserve(200);

    const int Nl = keypoints_left.size();
    const int Nr = keypoints_right.size();

    for (int iR = 0; iR < Nr; iR++) {
      const cv::KeyPoint &kp = keypoints_right[iR];
      const float &kpY = kp.pt.y;
      const float r = 2.0f * mpORBextractorLeft->GetScaleFactors()[keypoints_right[iR].octave];
      const int maxr = ceil(kpY + r);
      const int minr = floor(kpY - r);

      for (int yi = minr; yi <= maxr; yi++)
        vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = stereo_baseline_;
    const float minD = 0;
    const float maxD = stereo_baseline_f_ / minZ;

    // For each left keypoint search a match in the right image
    std::vector<std::pair<int, int>> vDistIdx;
    vDistIdx.reserve(Nl);

    for (int iL = 0; iL < Nl; iL++) {
      const cv::KeyPoint &kpL = keypoints_left[iL];
      const int &levelL = kpL.octave;
      const float &vL = kpL.pt.y;
      const float &uL = kpL.pt.x;

      const std::vector<std::size_t> &vCandidates = vRowIndices[vL];

      if (vCandidates.empty())
        continue;

      const float minU = uL - maxD;
      const float maxU = uL - minD;

      if (maxU < 0)
        continue;

      int bestDist = orb_th_high;
      size_t bestIdxR = 0;

      const cv::Mat &dL = descriptors_left.row(iL);

      // Compare descriptor to right keypoints
      for (size_t iC = 0; iC < vCandidates.size(); iC++) {
        const size_t iR = vCandidates[iC];
        const cv::KeyPoint &kpR = keypoints_right[iR];

        if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
          continue;

        const float &uR = kpR.pt.x;

        if (uR >= minU && uR <= maxU) {
          const cv::Mat &dR = descriptors_right.row(iR);
          const int dist = descriptorDistance(dL, dR);

          if (dist < bestDist) {
            bestDist = dist;
            bestIdxR = iR;
          }
        }
      }

      // Subpixel match by correlation
      if (bestDist < thOrbDist) {
        // coordinates in image pyramid at keypoint scale
        const float uR0 = keypoints_right[bestIdxR].pt.x;
        const float scaleFactor = orbExtractorInvScaleFactors[kpL.octave];
        const float scaleduL = round(kpL.pt.x * scaleFactor);
        const float scaledvL = round(kpL.pt.y * scaleFactor);
        const float scaleduR0 = round(uR0 * scaleFactor);

        // sliding window search
        const int w = 5;
        cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave]
                         .rowRange(scaledvL - w, scaledvL + w + 1)
                         .colRange(scaleduL - w, scaleduL + w + 1);

        int bestDist = INT_MAX;
        int bestincR = 0;
        const int L = 5;
        std::vector<float> vDists;
        vDists.resize(2 * L + 1);

        const float iniu = scaleduR0 + L - w;
        const float endu = scaleduR0 + L + w + 1;
        if (iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
          continue;

        for (int incR = -L; incR <= +L; incR++) {
          cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave]
                           .rowRange(scaledvL - w, scaledvL + w + 1)
                           .colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);

          float dist = cv::norm(IL, IR, cv::NORM_L1);
          if (dist < bestDist) {
            bestDist = dist;
            bestincR = incR;
          }

          vDists[L + incR] = dist;
        }

        if (bestincR == -L || bestincR == L)
          continue;

        // Sub-pixel match (Parabola fitting)
        const float dist1 = vDists[L + bestincR - 1];
        const float dist2 = vDists[L + bestincR];
        const float dist3 = vDists[L + bestincR + 1];

        const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

        if (deltaR < -1 || deltaR > 1)
          continue;

        // Re-scaled coordinate
        float bestuR = orbExtractorScaleFactors[kpL.octave] * ((float)scaleduR0 + (float)bestincR + deltaR);

        float disparity = (uL - bestuR);

        if (disparity >= minD && disparity < maxD) {
          if (disparity <= 0) {
            disparity = 0.01;
            bestuR = uL - 0.01;
          }
          depth[iL] = stereo_baseline_f_ / disparity;
          uRight[iL] = bestuR;

          matches_left_to_right.push_back(cv::DMatch(iL, bestIdxR, bestDist));

          vDistIdx.push_back(std::pair<int, int>(bestDist, iL));
        }
      }
    }

    sort(vDistIdx.begin(), vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size() / 2].first;
    const float thDist = 1.5f * 1.4f * median;

    for (int i = vDistIdx.size() - 1; i >= 0; i--) {
      if (vDistIdx[i].first < thDist)
        break;
      else {
        uRight[vDistIdx[i].second] = -1;
        depth[vDistIdx[i].second] = -1;
      }
    }
  }

  void loadEuroc() {
    std::cout << "loading images\n";
    std::string pathCam0 = eurocFolder_ + "/mav0/cam0/data";
    std::string pathCam1 = eurocFolder_ + "/mav0/cam1/data";

    // Loading image filenames and timestamps
    std::ifstream fTimes;
    std::cout << "opening timestamps file  " << eurocTimestampsFilename_ << "\n";
    fTimes.open(eurocTimestampsFilename_.c_str());
    if (!fTimes.is_open()) {
      std::cout << "could not open timestamps file\n";
      exit(1);
    }
    vTimestampsCam.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);
    while (!fTimes.eof()) {

      std::string s;
      std::getline(fTimes, s);
      std::cout << "timestamp: " << s << "\n";
      if (!s.empty()) {
        std::stringstream ss;
        ss << s;
        vstrImageLeft.push_back(pathCam0 + "/" + ss.str() + ".png");
        vstrImageRight.push_back(pathCam1 + "/" + ss.str() + ".png");
        double t;
        ss >> t;
        static double t_0 = t / 1e9;
        vTimestampsCam.push_back(t / 1e9 - t_0);
      }
    }
    dT_ = (vTimestampsCam[vTimestampsCam.size() - 1] - vTimestampsCam[0])/(vTimestampsCam.size());
    dTimeStamp_ = TimeStamp(dT_);
    nImages = vstrImageLeft.size();
    kMax_ = nImages;

    cv::Mat imLeft, imRight;
    cv::Mat imLeft_rect, imRight_rect;

    // Seq loop
    double t_resize = 0;
    double t_rect = 0;
    double t_track = 0;
    int num_rect = 0;

    // initial_component_.poses_.resize(nImages);
    // initial_component_.numPoses_= nImages;
    // initial_component_.numPoints_ = 0;

    measurements_.reserve(nImages);

    MotionModel_Odometry6d::TInput zero;
    MotionModel_Odometry6d::TInput::Vec u0;
    u0.setZero();
    zero.set(u0, 0);
    deadReckoning_pose_.resize(nImages);
    odometry_.resize(nImages, zero);

    for (int ni = 0; ni < nImages; ni++) {
      std::cout << "  loading image " << ni + 1 << "/" << nImages << "\n";

      std::vector<cv::KeyPoint> keypoints_left, keypoints_right;

      std::vector<ORBDescriptor> descriptors_left, descriptors_right;

      std::vector<float> uRight;
      std::vector<float> depth;
      std::vector<Eigen::Vector3d> point_camera_frame;

      std::vector<cv::DMatch> matches_left_to_right;

      // Read left and right images from file
      imLeft = cv::imread(vstrImageLeft[ni], cv::IMREAD_UNCHANGED);   //,cv::IMREAD_UNCHANGED);
      imRight = cv::imread(vstrImageRight[ni], cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);

      if (imLeft.empty()) {
        std::cerr << std::endl << "Failed to load image at: " << std::string(vstrImageLeft[ni]) << std::endl;
        exit(1);
      }

      if (imRight.empty()) {
        std::cerr << std::endl << "Failed to load image at: " << std::string(vstrImageRight[ni]) << std::endl;
        exit(1);
      }

      double tframe = vTimestampsCam[ni];

      cv::remap(imLeft, imLeft_rect, camera_parameters_[0].M1, camera_parameters_[0].M2, cv::INTER_LINEAR);
      cv::remap(imRight, imRight_rect, camera_parameters_[1].M1, camera_parameters_[1].M2, cv::INTER_LINEAR);

      std::vector<int> vLapping_left = {0, 0};
      std::vector<int> vLapping_right = {0, 0};
      cv::Mat mask_left, mask_right;

      //		std::thread threadLeft(&ORB_SLAM3::ORBextractor::extract,
      //				mpORBextractorLeft, &imLeft_rect, &mask_left,
      //				&keypoints_left,
      //				&descriptors_left,
      //				&vLapping_left);
      //		std::thread threadRight(&ORB_SLAM3::ORBextractor::extract,
      //				mpORBextractorRight, &imRight_rect, &mask_right,
      //				&keypoints_right,
      //				&descriptors_right,
      //				&vLapping_right);
      //		threadLeft.join();
      //		threadRight.join();

      cv::Mat desc_left, desc_right;
      ORB_SLAM3::ORBextractor::extract(mpORBextractorLeft, &imLeft_rect, &mask_left, &keypoints_left, &desc_left, &vLapping_left);
      ORB_SLAM3::ORBextractor::extract(mpORBextractorRight, &imRight_rect, &mask_right, &keypoints_right, &desc_right,
                                       &vLapping_right);
      computeStereoMatches(keypoints_left, keypoints_right, desc_left, desc_right, matches_left_to_right);
      stereoMatchesToMeasurments(keypoints_left, keypoints_right, desc_left, desc_right, matches_left_to_right, measurements_, tframe);

      // plot stereo matches
      cv::Mat imLeftKeys, imRightKeys, imMatches;
      cv::Scalar kpColor = cv::Scalar(255, 0, 0);

      cv::drawMatches(imLeft_rect, keypoints_left, imRight_rect, keypoints_right, matches_left_to_right, imMatches);

      cv::drawKeypoints(imRight_rect, keypoints_right, imRightKeys, kpColor);
      cv::drawKeypoints(imLeft_rect, keypoints_left, imLeftKeys, kpColor);
      // cv::imshow("matches", imMatches);

      cv::waitKey(1); // Wait for a keystroke in the window
      std::cout << ni + 1 << "/" << nImages << "                                   \r";
    }
    std::cout <<  "loaded images\n";
    std::cout << "\n";
  }

  /** Run the simulator */
  void run() {

    printf("Running simulation\n\n");

    MotionModel_Odometry6d::TState zero_pose;
#ifdef _PERFTOOLS_CPU
    std::string perfCPU_file = logDirPrefix_ + "rbphdslam2dSim_cpu.prof";
    ProfilerStart(perfCPU_file.data());
#endif
#ifdef _PERFTOOLS_HEAP
    std::string perfHEAP_file = logDirPrefix_ + "rbphdslam2dSim_heap.prof";
    HeapProfilerStart(perfHEAP_file.data());
#endif

    //////// Initialization at first timestep //////////

    if (!logResultsToFile_) {
      std::cout << "Note: results are NOT being logged to file (see config xml file)\n";
    }
    FILE *pParticlePoseFile;
    if (logResultsToFile_) {
      std::string filenameParticlePoseFile(logDirPrefix_);
      filenameParticlePoseFile += "particlePose.dat";
      pParticlePoseFile = fopen(filenameParticlePoseFile.data(), "w");
    }
    FILE *pLandmarkEstFile;
    if (logResultsToFile_) {
      std::string filenameLandmarkEstFile(logDirPrefix_);
      filenameLandmarkEstFile += "landmarkEst.dat";
      pLandmarkEstFile = fopen(filenameLandmarkEstFile.data(), "w");
    }
    MotionModel_Odometry6d::TState x_i;
    int zIdx = 0;

    if (logResultsToFile_) {
      for (int i = 0; i < pFilter_->getParticleCount(); i++) {
        x_i = *(pFilter_->getParticleSet()->at(i));
        fprintf(pParticlePoseFile, "%f   %d   %f   %f   %f   %f   %f   %f   %f   1.0\n", 0.0, i, x_i.get(0), x_i.get(1),
                x_i.get(2), x_i.get(3), x_i.get(4), x_i.get(5), x_i.get(6));
      }
    }

    /////////// Run simulator from k = 1 to kMax_ /////////

    TimeStamp time;

    for (int k = 1; k < kMax_; k++) {

      time += dTimeStamp_;

      if (k % 100 == 0 || k == kMax_ - 1) {
        float progressPercent = float(k + 1) / float(kMax_);
        int progressBarW = 50;
        struct winsize ws;
        if (ioctl(1, TIOCGWINSZ, &ws) >= 0)
          progressBarW = ws.ws_col - 30;
        int progressPos = progressPercent * progressBarW;
        if (progressBarW >= 50) {
          std::cout << "[";
          for (int i = 0; i < progressBarW; i++) {
            if (i < progressPos)
              std::cout << "=";
            else if (i == progressPos)
              std::cout << ">";
            else
              std::cout << " ";
          }
          std::cout << "] ";
        }
        std::cout << "k = " << k << " (" << int(progressPercent * 100.0) << " %)\r";
        std::cout.flush();
      }
      if (k == kMax_ - 1)
        std::cout << std::endl << std::endl;

#ifdef _PERFTOOLS_HEAP
      if (k % 20 == 0)
        HeapProfilerDump("Timestep interval dump");
#endif

      ////////// Prediction Step //////////

      // configure robot motion model ( not necessary since in simulation, timesteps are constant)
      // MotionModel_Odometry2d::TState::Mat Q;
      // Q << vardx_, 0, 0, 0, vardy_, 0, 0, 0, vardz_;
      // Q *= (pNoiseInflation_ * dt * dt);
      // pFilter_->getProcessModel()->setNoise(Q);

      // configure landmark process model ( not necessary since in simulation, timesteps are constant)
      // Landmark2d::Mat Q_lm;
      // Q_lm << varlmx_, 0, 0, varlmy_;
      // Q_lm = Q_lm * dt * dt;
      // pFilter_->getLmkProcessModel()->setNoise(Q_lm);

      pFilter_->predict(odometry_[k], dTimeStamp_);

      if (k <= 2) {
        for (int i = 0; i < nParticles_; i++)
          pFilter_->setParticlePose(i, zero_pose);
      }

      // Prepare measurement vector for update
      std::vector<MeasurementModel_3D_stereo_orb::TMeasurement> Z;
      TimeStamp time = measurements_[zIdx].getTime();
      std::cout << "time: " << time.getTimeAsDouble() << "  " << measurements_[zIdx].getTime().getTimeAsDouble() << "\n";
      std::cout << "equals: " << (measurements_[zIdx].getTime() == time) << "\n";
      while (measurements_[zIdx].getTime() == time) {

        Z.push_back(measurements_[zIdx]);
        zIdx++;
        if (zIdx >= measurements_.size())
          break;
      }
      std::cout << "number of measurements: " << Z.size() << "\n";

      ////////// Update Step //////////
      pFilter_->update(Z);

      // Log particle poses
      int i_w_max = 0;
      double w_max = 0;
      if (logResultsToFile_) {
        for (int i = 0; i < pFilter_->getParticleCount(); i++) {
          x_i = *(pFilter_->getParticleSet()->at(i));
          double w = pFilter_->getParticleSet()->at(i)->getWeight();
          if (w > w_max) {
            i_w_max = i;
            w_max = w;
          }
          fprintf(pParticlePoseFile, "%f   %d   %f   %f   %f   %f   %f   %f   %f   %f\n", time.getTimeAsDouble(), i, x_i.get(0),
                  x_i.get(1), x_i.get(2), x_i.get(3), x_i.get(4), x_i.get(5), x_i.get(6), w);
        }
        fprintf(pParticlePoseFile, "\n");
      }

      // Log landmark estimates
      if (logResultsToFile_) {

        int mapSize = pFilter_->getGMSize(i_w_max);
        for (int m = 0; m < mapSize; m++) {
          MeasurementModel_3D_stereo_orb::TLandmark::Vec u;
          MeasurementModel_3D_stereo_orb::TLandmark::Mat S;
          double w;
          pFilter_->getLandmark(i_w_max, m, u, S, w);

          fprintf(pLandmarkEstFile, "%f   %d   ", time.getTimeAsDouble(), i_w_max);
          fprintf(pLandmarkEstFile, "%f   %f   %f      ", u(0), u(1), u(2));
          fprintf(pLandmarkEstFile, "%f   %f   %f   %f   %f   %f", S(0, 0), S(0, 1), S(0, 2), S(1, 1), S(1, 2), S(2, 2));
          fprintf(pLandmarkEstFile, "   %f\n", w);
        }
      }

      // Visualization
      if (use_ros_gui_) {


	marker_pub_->publish(makeRosMarkerArray());
	landmark_cloud_pub_->publish(makeRosPointcloud());
	

	
      
      }
    }

#ifdef _PERFTOOLS_HEAP
    HeapProfilerStop();
#endif
#ifdef _PERFTOOLS_CPU
    ProfilerStop();
#endif

    std::cout << "Elapsed Timing Information [nsec]\n";
    std::cout << std::setw(15) << std::left << "Prediction" << std::setw(15) << std::setw(6) << std::right
              << "wall:" << std::setw(15) << pFilter_->getTimingInfo()->predict_wall << std::setw(6) << std::right
              << "cpu:" << std::setw(15) << pFilter_->getTimingInfo()->predict_cpu << std::endl;
    std::cout << std::setw(15) << std::left << "Map Update" << std::setw(15) << std::setw(6) << std::right
              << "wall:" << std::setw(15) << pFilter_->getTimingInfo()->mapUpdate_wall << std::setw(6) << std::right
              << "cpu:" << std::setw(15) << pFilter_->getTimingInfo()->mapUpdate_cpu << std::endl;
    std::cout << std::setw(15) << std::left << "Map Update (KF)" << std::setw(15) << std::setw(6) << std::right
              << "wall:" << std::setw(15) << pFilter_->getTimingInfo()->mapUpdate_kf_wall << std::setw(6) << std::right
              << "cpu:" << std::setw(15) << pFilter_->getTimingInfo()->mapUpdate_kf_cpu << std::endl;
    std::cout << std::setw(15) << std::left << "Weighting" << std::setw(15) << std::setw(6) << std::right
              << "wall:" << std::setw(15) << pFilter_->getTimingInfo()->particleWeighting_wall << std::setw(6) << std::right
              << "cpu:" << std::setw(15) << pFilter_->getTimingInfo()->particleWeighting_cpu << std::endl;
    std::cout << std::setw(15) << std::left << "Map Merge" << std::setw(15) << std::setw(6) << std::right
              << "wall:" << std::setw(15) << pFilter_->getTimingInfo()->mapMerge_wall << std::setw(6) << std::right
              << "cpu:" << std::setw(15) << pFilter_->getTimingInfo()->mapMerge_cpu << std::endl;
    std::cout << std::setw(15) << std::left << "Map Prune" << std::setw(15) << std::setw(6) << std::right
              << "wall:" << std::setw(15) << pFilter_->getTimingInfo()->mapPrune_wall << std::setw(6) << std::right
              << "cpu:" << std::setw(15) << pFilter_->getTimingInfo()->mapPrune_cpu << std::endl;
    std::cout << std::setw(15) << std::left << "Resampling" << std::setw(15) << std::setw(6) << std::right
              << "wall:" << std::setw(15) << pFilter_->getTimingInfo()->particleResample_wall << std::setw(6) << std::left
              << std::right << "cpu:" << std::setw(15) << pFilter_->getTimingInfo()->particleResample_cpu << std::endl;
    std::cout << std::setw(15) << std::left << "Total" << std::setw(15) << std::setw(6) << std::right << "wall:" << std::setw(15)
              << pFilter_->getTimingInfo()->predict_wall + pFilter_->getTimingInfo()->mapUpdate_wall +
                     pFilter_->getTimingInfo()->particleWeighting_wall + pFilter_->getTimingInfo()->mapMerge_wall +
                     pFilter_->getTimingInfo()->mapPrune_wall + pFilter_->getTimingInfo()->particleResample_wall
              << std::setw(6) << std::right << "cpu:" << std::setw(15)
              << pFilter_->getTimingInfo()->predict_cpu + pFilter_->getTimingInfo()->mapUpdate_cpu +
                     pFilter_->getTimingInfo()->particleWeighting_cpu + pFilter_->getTimingInfo()->mapMerge_cpu +
                     pFilter_->getTimingInfo()->mapPrune_cpu + pFilter_->getTimingInfo()->particleResample_cpu
              << std::endl;

    if (logTimingToFile_) {
      std::ofstream timingFile((logDirPrefix_ + "timing.dat").data());
      timingFile << "Elapsed Timing Information [nsec]\n";
      timingFile << std::setw(15) << std::left << "Prediction" << std::setw(15) << std::setw(6) << std::right
                 << "wall:" << std::setw(15) << pFilter_->getTimingInfo()->predict_wall << std::setw(6) << std::right
                 << "cpu:" << std::setw(15) << pFilter_->getTimingInfo()->predict_cpu << std::endl;
      timingFile << std::setw(15) << std::left << "Map Update" << std::setw(15) << std::setw(6) << std::right
                 << "wall:" << std::setw(15) << pFilter_->getTimingInfo()->mapUpdate_wall << std::setw(6) << std::right
                 << "cpu:" << std::setw(15) << pFilter_->getTimingInfo()->mapUpdate_cpu << std::endl;
      timingFile << std::setw(15) << std::left << "Map Update (KF)" << std::setw(15) << std::setw(6) << std::right
                 << "wall:" << std::setw(15) << pFilter_->getTimingInfo()->mapUpdate_kf_wall << std::setw(6) << std::right
                 << "cpu:" << std::setw(15) << pFilter_->getTimingInfo()->mapUpdate_kf_cpu << std::endl;
      timingFile << std::setw(15) << std::left << "Weighting" << std::setw(15) << std::setw(6) << std::right
                 << "wall:" << std::setw(15) << pFilter_->getTimingInfo()->particleWeighting_wall << std::setw(6) << std::right
                 << "cpu:" << std::setw(15) << pFilter_->getTimingInfo()->particleWeighting_cpu << std::endl;
      timingFile << std::setw(15) << std::left << "Map Merge" << std::setw(15) << std::setw(6) << std::right
                 << "wall:" << std::setw(15) << pFilter_->getTimingInfo()->mapMerge_wall << std::setw(6) << std::right
                 << "cpu:" << std::setw(15) << pFilter_->getTimingInfo()->mapMerge_cpu << std::endl;
      timingFile << std::setw(15) << std::left << "Map Prune" << std::setw(15) << std::setw(6) << std::right
                 << "wall:" << std::setw(15) << pFilter_->getTimingInfo()->mapPrune_wall << std::setw(6) << std::right
                 << "cpu:" << std::setw(15) << pFilter_->getTimingInfo()->mapPrune_cpu << std::endl;
      timingFile << std::setw(15) << std::left << "Resampling" << std::setw(15) << std::setw(6) << std::right
                 << "wall:" << std::setw(15) << pFilter_->getTimingInfo()->particleResample_wall << std::setw(6) << std::left
                 << std::right << "cpu:" << std::setw(15) << pFilter_->getTimingInfo()->particleResample_cpu << std::endl;
      timingFile << std::setw(15) << std::left << "Total" << std::setw(15) << std::setw(6) << std::right
                 << "wall:" << std::setw(15)
                 << pFilter_->getTimingInfo()->predict_wall + pFilter_->getTimingInfo()->mapUpdate_wall +
                        pFilter_->getTimingInfo()->particleWeighting_wall + pFilter_->getTimingInfo()->mapMerge_wall +
                        pFilter_->getTimingInfo()->mapPrune_wall + pFilter_->getTimingInfo()->particleResample_wall
                 << std::setw(6) << std::right << "cpu:" << std::setw(15)
                 << pFilter_->getTimingInfo()->predict_cpu + pFilter_->getTimingInfo()->mapUpdate_cpu +
                        pFilter_->getTimingInfo()->particleWeighting_cpu + pFilter_->getTimingInfo()->mapMerge_cpu +
                        pFilter_->getTimingInfo()->mapPrune_cpu + pFilter_->getTimingInfo()->particleResample_cpu
                 << std::endl;
      timingFile.close();
    }

    if (logResultsToFile_) {
      fclose(pParticlePoseFile);
      fclose(pLandmarkEstFile);
    }
  }

private:
  const char *cfgFileName_;

  int kMax_;             /**< number of timesteps */
  double dT_;            /**< duration of timestep in seconds */
  TimeStamp dTimeStamp_; /**< duration of timestep in timestamp */

  // Trajectory
  int nSegments_;
  double max_dx_;
  double max_dy_;
  double max_dz_;
  double max_dqx_;
  double max_dqy_;
  double max_dqz_;
  double max_dqw_;
  double min_dx_;
  double vardx_;
  double vardy_;
  double vardz_;
  double vardqx_;
  double vardqy_;
  double vardqz_;
  double vardqw_;
  // std::vector<MotionModel_Odometry6d::TInput> groundtruth_displacement_;
  // std::vector<MotionModel_Odometry6d::TState> groundtruth_pose_;
  std::vector<MotionModel_Odometry6d::TInput> odometry_;
  std::vector<MotionModel_Odometry6d::TState> deadReckoning_pose_;

  // Landmarks
  int nLandmarks_;
  // std::vector<MeasurementModel_3D_stereo_orb::TLandmark> groundtruth_landmark_;
  double varlmx_;
  double varlmy_;
  double varlmz_;
  std::vector<double> lmkFirstObsTime_;

  // Range-Bearing Measurements
  double rangeLimitMax_;
  double rangeLimitMin_;
  double rangeLimitBuffer_;
  double Pd_;
  double c_;
  double varzx_;
  double varzy_;
  double varzz_;
  std::vector<MeasurementModel_3D_stereo_orb::TMeasurement> measurements_;

  // Filters
  KalmanFilter<StaticProcessModel<Landmark3d>, MeasurementModel_3D_stereo_orb> kf_;
  RBPHDFilter<MotionModel_Odometry6d, StaticProcessModel<Landmark3d>, MeasurementModel_3D_stereo_orb,
              KalmanFilter<StaticProcessModel<Landmark3d>, MeasurementModel_3D_stereo_orb>> *pFilter_;
  int nParticles_;
  double pNoiseInflation_;
  double zNoiseInflation_;
  double innovationRangeThreshold_;
  double innovationBearingThreshold_;
  double birthGaussianWeight_;
  double newGaussianCreateInnovMDThreshold_;
  double importanceWeightingMeasurementLikelihoodMDThreshold_;
  double importanceWeightingEvalPointGuassianWeight_;
  double effNParticleThreshold_;
  int minUpdatesBeforeResample_;
  double gaussianMergingThreshold_;
  double gaussianMergingCovarianceInflationFactor_;
  double gaussianPruningThreshold_;
  int importanceWeightingEvalPointCount_;
  bool useClusterProcess_;

  bool logResultsToFile_;
  bool logTimingToFile_;

  struct CameraParams {
    double fx;
    double fy;
    double cx;
    double cy;
    double k1;
    double k2;
    double p1;
    double p2;

    cv::Size originalImSize, newImSize;
    cv::Mat opencv_distort_coeffs, opencv_calibration;
    cv::Mat M1, M2;

    cv::Mat cv_c0_to_camera;

    Eigen::MatrixXd cv_c0_to_camera_eigen;
  };
  std::vector<CameraParams> camera_parameters_;

  double stereo_baseline_;
  double stereo_baseline_f_;
  double stereo_init_max_depth_;

  // euroc dataset

  // filenames
  std::vector<std::string> vstrImageLeft;
  std::vector<std::string> vstrImageRight;
  // timestamps

  std::vector<double> vTimestampsCam;
  int nImages;

		// ORB params
		struct ORBExtractor {
			int nFeatures;
			double scaleFactor;
			int nLevels;
			int iniThFAST;
			int minThFAST;

		} orb_extractor;

  ORB_SLAM3::ORBextractor *mpORBextractorLeft;
  ORB_SLAM3::ORBextractor *mpORBextractorRight;
  std::vector<float> orbExtractorInvScaleFactors;
  std::vector<float> orbExtractorScaleFactors;
  std::string eurocFolder_;
  std::string eurocTimestampsFilename_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr landmark_cloud_pub_;
  




public:
  rclcpp::Node::SharedPtr visualization_node_;
  // 3D visualization
  bool use_ros_gui_;
  std::string logDirPrefix_;
};

int main(int argc, char *argv[]) {

  rfs::initializeGaussianGenerators();
  Simulator_RBPHDSLAM_6d sim;

  int seed = time(NULL);
  srand(seed);
  int trajNum = rand();
  std::string cfgFileName;
  bool printHelp = false;
  argparse::ArgumentParser parser("This is a test program for argparse");
  parser.add_argument("-h", "--help").help("produce this help message").store_into(printHelp);
  parser.add_argument("-c", "--cfg")
      .help("configuration xml file")
      .default_value("cfg/rbphdslam2dSim.yaml")
      .store_into(cfgFileName);
  parser.add_argument("-t", "--trajectory").help("trajectory number (default: a random integer)").store_into(trajNum);
  parser.add_argument("-s", "--seed")
      .help("random seed for running the simulation (default: based on current system time)")
      .store_into(seed);
  try {
    parser.parse_args(argc, argv);
  } catch (const std::runtime_error &e) {
    std::cout << e.what() << std::endl;
    std::cout << parser;
    return 1;
  }

  if (printHelp) {
    std::cout << parser;
    return 1;
  }

  std::cout << "Configuration file: " << cfgFileName << std::endl;
  if (!sim.readConfigFile(cfgFileName.data())) {
    return -1;
  }

  if (sim.use_ros_gui_) {
    rclcpp::init(argc, argv);
    sim.visualization_node_ = std::make_shared<rclcpp::Node>("visualization");
  }

  std::cout << "Trajectory: " << trajNum << std::endl;

  sim.loadEuroc();
  sim.exportSimData();
  sim.setupRBPHDFilter();

  if (parser.is_used("seed")) {
    std::cout << "Simulation random seed manually set to: " << seed << std::endl;
  } else {
    std::cout << "Simulation random seed set to: " << seed << std::endl;
  }
  srand48(seed);

  // boost::timer::auto_cpu_timer *timer = new boost::timer::auto_cpu_timer(6, "Simulation run time: %ws\n");

  sim.run();

  // std::cout << "mem use: " << MemProfile::getCurrentRSS() << "(" << MemProfile::getPeakRSS() << ")\n";
  // delete timer;

  return 0;
}
