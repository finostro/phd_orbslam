#include "Landmark.hpp"
#include "visualization_msgs/msg/marker.hpp"

namespace rfs{
visualization_msgs::msg::Marker to_marker(const Landmark3d &lm)
{
  visualization_msgs::msg::Marker marker;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.header.frame_id = "map";
  marker.ns = "landmarks";
  marker.type = visualization_msgs::msg::Marker::SPHERE;
  marker.pose.position.x = lm.get()(0);
  marker.pose.position.y = lm.get()(1);
  marker.pose.position.z = lm.get()(2);
  
  marker.points.resize(1);
  marker.points[0].x= 0.;
  marker.points[0].y= 0.;
  marker.points[0].z= 0.;

  marker.color.a = 1.0;
  marker.color.g = 0.9;

  auto cov = lm.getCov();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(cov);
  if (eigensolver.info() != Eigen::Success) {
    std::cerr << "bad covariance eigenvalues\n";

  } else {
    Eigen::Quaterniond q(eigensolver.eigenvectors());

    marker.pose.orientation.w = q.w();
    marker.pose.orientation.x = q.x();
    marker.pose.orientation.y = q.y();
    marker.pose.orientation.z = q.z();


    marker.scale.x = sqrt(eigensolver.eigenvalues()(0));
    marker.scale.y = sqrt(eigensolver.eigenvalues()(1));
    marker.scale.z = sqrt(eigensolver.eigenvalues()(2));
  }

  return marker;
}
}
