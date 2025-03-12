# Use ROS 2 Humble as base image
FROM ros:humble

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_WS=/ros_ws

# Install essential dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up the workspace
WORKDIR $ROS_WS

# Clone your repository
RUN git clone https://github.com/PietDol/FI-CBF.git $ROS_WS/src/FI-CBF

# Install ROS 2 dependencies using rosdep
RUN apt-get update && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

# Build the ROS 2 package
RUN bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Set up entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash"]