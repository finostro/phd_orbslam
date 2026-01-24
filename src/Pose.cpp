
#include "Pose.hpp"

namespace rfs{
geometry_msgs::msg::PoseStamped to_msg(const Pose6d &x)
{

  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "map";
  pose.pose.position.x = x.get()(0);
  pose.pose.position.y = x.get()(1);
  pose.pose.position.z = x.get()(2);
  Eigen::Quaterniond q(x.getRot());
  pose.pose.orientation.x = q.x();
  pose.pose.orientation.y = q.y();
  pose.pose.orientation.z = q.z();
  pose.pose.orientation.w = q.w();
  return pose;
}
}
