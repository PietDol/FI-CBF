#!/bin/bash
set -e

# Source ROS 2 and workspace
source /opt/ros/humble/setup.bash
source /ros_ws/install/setup.bash

exec "$@"