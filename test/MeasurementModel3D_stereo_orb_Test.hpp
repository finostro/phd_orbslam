// Test for classes derived from MeasurementModel
// Keith Leung 2013

#include "Landmark.hpp"
#include "Measurement.hpp"
#include "Pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "measurement_models/MeasurementModel_3D_stereo_orb.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "gtest/gtest.h"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/AngleAxis.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <memory>
#include <rclcpp/duration.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <visualization_msgs/msg/marker.hpp>

/**
 * \class MeasurementModel_3D_stereo_orbTest
 * \brief Unit testing fixture for MeasurementModel derived classes
 * \author Keith Leung
 */

class MeasurementModel_3D_stereo_orbTest : public ::testing::Test {

protected:
  /** Constructor for setting up each test */
  MeasurementModel_3D_stereo_orbTest() : model_(100.0, 100.0, 100.0)
  {

    auto camera_params =
        std::make_shared<gtsam::Cal3_S2Stereo>(FX, FY, SKEW, CX, CY, BASELINE);

    gtsam::StereoCamera stereocamera(gtsam::Pose3(), camera_params);
    model_.config.camera.camera = stereocamera;
    model_.config.camera.bounds.mnMinX = 0;
    model_.config.camera.bounds.mnMaxX = WIDTH;
    model_.config.camera.bounds.mnMinY = 0;
    model_.config.camera.bounds.mnMaxY = HEIGHT;
    model_.config.camera.viewingCosLimit = 0.3;

    int argc = 1;
    auto fake_arg = "test";
    
    rclcpp::init(argc, &fake_arg);
    node_ = std::make_shared<rclcpp::Node>("testing_node");
    marker_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>("debug_markers", rclcpp::SystemDefaultsQoS());
    pose_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>("pose", rclcpp::SystemDefaultsQoS());
    node_->get_clock()->sleep_for(rclcpp::Duration::from_seconds(.05));

  }

  /** Destructor */
  virtual ~MeasurementModel_3D_stereo_orbTest() {
    rclcpp::shutdown();
  }

  /** Setup -- called after constructor before each test */
  virtual void SetUp() override {}

  /** Teardown -- called after each test but before destructor */
  virtual void TearDown() override {}

  // Additional objects to declare //
  rfs::MeasurementModel_3D_stereo_orb model_;
  std::shared_ptr<rclcpp::Node> node_;
  std::shared_ptr<rclcpp::Publisher<visualization_msgs::msg::Marker>> marker_pub_;
  std::shared_ptr<rclcpp::Publisher<geometry_msgs::msg::PoseStamped>> pose_pub_;


  static constexpr double WIDTH = 1080;
  static constexpr double HEIGHT = 720;

  static constexpr double FX = 450;
  static constexpr double FY = 450;
  static constexpr double SKEW = 0.0;
  static constexpr double CX = WIDTH / 2.0;
  static constexpr double CY = HEIGHT / 2.0;
  static constexpr double BASELINE = 0.1;
};

////////// Test Cases //////////

TEST_F(MeasurementModel_3D_stereo_orbTest, DirectMeasure0) {
  using namespace rfs;
  Pose6d pose;
  pose[6] = 1.0;
  Landmark3d lm;
  Measurement3d z;
  bool measured = model_.measure(pose, lm, z);
  EXPECT_FALSE(measured);
}

TEST_F(MeasurementModel_3D_stereo_orbTest, DirectMeasure1) {
  using namespace rfs;
  Pose6d pose;
  pose[6] = 1.0;
  Landmark3d lm;
  lm[0] = 0.0;
  lm[1] = 0.0;
  lm[2] = 1.0;
  Measurement3d z;
  Measurement3d expected_z;
  expected_z[0] = WIDTH/2;
  expected_z[1] = WIDTH/2 -BASELINE*FX;
  expected_z[2] = HEIGHT/2;

  bool measured = model_.measure(pose, lm, z);
  EXPECT_TRUE(measured);
  EXPECT_DOUBLE_EQ(expected_z[0], z[0]);
  EXPECT_DOUBLE_EQ(expected_z[1], z[1]);
  EXPECT_DOUBLE_EQ(expected_z[2], z[2]);
}

TEST_F(MeasurementModel_3D_stereo_orbTest, InverseMeasure1) {
  using namespace rfs;
  Pose6d pose;
  pose[6] = 1.0;
  Landmark3d expected_lm;
  expected_lm[0] = 0.0;
  expected_lm[1] = 0.0;
  expected_lm[2] = 1.0;
  Landmark3d lm;
  Measurement3d z;
  z[0] = WIDTH/2;
  z[1] = WIDTH/2 -BASELINE*FX;
  z[2] = HEIGHT/2;
  model_.inverseMeasure(pose, z, lm);
  EXPECT_DOUBLE_EQ(expected_lm[0], lm[0]);
  EXPECT_DOUBLE_EQ(expected_lm[1], lm[1]);
  EXPECT_DOUBLE_EQ(expected_lm[2], lm[2]);
}

TEST_F(MeasurementModel_3D_stereo_orbTest, InverseMeasure2) {
  using namespace rfs;
  for( double angle = 0; angle < 20*3.1415 ; angle += 0.01){
  Pose6d pose;
  pose[0] = 1.0;
  pose[1] = 1.0;
  
  pose.setRot(Eigen::Quaterniond(Eigen::AngleAxisd(angle,Eigen::Vector3d(0,0,1))).coeffs());
  Landmark3d expected_lm;
  expected_lm[0] = -1.0;
  expected_lm[1] = 0.0;
  expected_lm[2] = 4.0;
  Landmark3d lm;
  Measurement3d z;
  z[0] = -1./4*FX+WIDTH/2;
  z[1] = -1./4*FX+WIDTH/2 -BASELINE*FX/4.0;
  z[2] = HEIGHT/2;
  model_.inverseMeasure(pose, z, lm);
  // EXPECT_DOUBLE_EQ(expected_lm[0], lm[0]);
  // EXPECT_DOUBLE_EQ(expected_lm[1], lm[1]);
  // EXPECT_DOUBLE_EQ(expected_lm[2], lm[2]);
  // for(int i = 0 ; i<30; i++){
    auto marker = to_marker(lm);
     
  auto pose_msg = to_msg(pose);
  pose_msg.pose = pose.toMsg();
  marker_pub_->publish(marker);
  pose_pub_->publish(pose_msg);
  node_->get_clock()->sleep_for(rclcpp::Duration::from_seconds(0.03));
  }
  // }
}

TEST_F(MeasurementModel_3D_stereo_orbTest, DirectJacobian1) {
  using namespace rfs;
  Pose6d pose;
  pose[6] = 1.0;
  Landmark3d lm;
  lm[0] = 0.0;
  lm[1] = 0.0;
  lm[2] = 1.0;
  Measurement3d z;
  Measurement3d expected_z;
  MeasurementModel_3D_stereo_orb::TJacobianPose pose_jk;
  expected_z[0] = WIDTH/2;
  expected_z[1] = WIDTH/2 -BASELINE*FX;
  expected_z[2] = HEIGHT/2;
  bool measured = model_.measure(pose, lm, z,NULL, &pose_jk);
  EXPECT_TRUE(measured);
  EXPECT_DOUBLE_EQ(expected_z[0], z[0]);
  EXPECT_DOUBLE_EQ(expected_z[1], z[1]);
  EXPECT_DOUBLE_EQ(expected_z[2], z[2]);
}

// Test MeasurementModel_3D_stereo_orbTest constructor, and set / get functions
TEST_F(MeasurementModel_3D_stereo_orbTest, orbModelConstructorTest) {
  Eigen::Matrix3d expected_cov;
  expected_cov << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
  Eigen::Matrix3d cov_out1;
  model_.getNoise(cov_out1);
  EXPECT_EQ(cov_out1, expected_cov);
}

// Test MeasurementModel_3D_stereo_orbTest constructor, and set / get functions
// TEST_F(MeasurementModel_3D_stereo_orbTest, rangeBearingModelPredictTest){
//
//   RangeBearingModel model(1, 1);
//   RangeBearingModel::TPose x;
//   Eigen::Vector3d xPose;
//   RangeBearingModel::TLandmark m;
//   Eigen::Vector2d mPos;
//   Eigen::Matrix2d mCov;
//   RangeBearingModel::TMeasurement z;
//   Eigen::Vector2d zVec;
//   Eigen::Matrix2d zCov;
//   Eigen::Matrix2d jacobian;
//   double t;
//
//   xPose << 0, 0, 0;
//   x.set(xPose);
//   mPos << -1, -1;
//   mCov << 1, 0, 0, 1;
//   m.set(mPos, mCov);
//   model.measure( x, m, z , &jacobian);
//   z.get(zVec, zCov, t);
//   EXPECT_EQ( sqrt(2), zVec(0) );
//   EXPECT_EQ( -0.75 * PI , zVec(1) );
// }

// TEST_F(MeasurementModel_3D_stereo_orbTest, rangeBearingModelInvPredictTest){
//
//   RangeBearingModel model(1, 1);
//   RangeBearingModel::TPose x;
//   Eigen::Vector3d xPose;
//   RangeBearingModel::TLandmark m;
//   Eigen::Vector2d mPos;
//   Eigen::Matrix2d mCov;
//   RangeBearingModel::TMeasurement z;
//   Eigen::Vector2d zVec;
//   Eigen::Matrix2d zCov;
//   double t;
//
//   xPose << 0, 0, 0;
//   x.set(xPose);
//   zVec << sqrt(2) , -0.75 * PI;
//   zCov << 1, 0, 0, 1;
//   z.set( zVec, zCov, 0.4565);
//   model.inverseMeasure(x, z, m);
//   m.get(mPos, mCov);
//   EXPECT_DOUBLE_EQ(-1, mPos(0));
//   EXPECT_DOUBLE_EQ(-1, mPos(1));
//
//   // Is there a good way of checking mCov?
//
// }
