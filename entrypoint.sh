#!/bin/bash
set -e

# Source ROS 2 and workspace
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source /ros_ws/install/setup.bash || true" >> ~/.bashrc
source ~/.bashrc

exec "$@"