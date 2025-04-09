# Use ROS 2 Humble as base image (Ubuntu 22.04)
FROM ros:humble

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_WS=/ros_ws
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

# Install essential dependencies for pyenv
RUN apt-get update && apt-get install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
    python3-pip python3-venv python3-dev \
    python3-colcon-common-extensions \
    git && \
    rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl https://pyenv.run | bash

# Ensure pyenv is initialized in every shell
RUN echo 'eval "$(pyenv init --path)"' >> /root/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> /root/.bashrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> /root/.bashrc

# Reload shell to recognize pyenv (for Docker RUN commands)
SHELL ["/bin/bash", "-c"]

# Install Python 3.10.8 via pyenv (following cbfpy official setup)
RUN pyenv install 3.10.8 && pyenv global 3.10.8

# Verify Python installation
RUN python --version  # Should print Python 3.10.8

# Set up pyenv virtual environment (following cbfpy instructions)
RUN pyenv virtualenv 3.10.8 cbfpy-env

# ✅ Use `pyenv exec` Instead of `pyenv activate`
RUN pyenv exec pip install --upgrade pip setuptools wheel

# Set up the workspace
WORKDIR $ROS_WS

# Clone the FI-CBF repository
RUN git clone https://github.com/PietDol/FI-CBF.git $ROS_WS/src/FI-CBF

# Clone and install CBFpy inside the ROS workspace
WORKDIR /ros_ws/src
RUN git clone https://github.com/danielpmorton/cbfpy.git

# Set the working directory to cbfpy
WORKDIR /ros_ws/src/cbfpy

# Install dependencies inside the virtual environment
# RUN pyenv exec pip install -r requirements.txt

# Install cbfpy inside the virtual environment in editable mode
RUN pyenv exec pip install -e ".[examples]"

# Return to the workspace
WORKDIR /ros_ws

# Install ROS 2 dependencies using rosdep
RUN apt-get update && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

# Build the ROS 2 package
RUN bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Modify entrypoint script to activate pyenv
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash"]
