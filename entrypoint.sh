#!/bin/bash
set -e

# Source ROS 2 and workspace
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source /ros_ws/install/setup.bash || true" >> ~/.bashrc
source ~/.bashrc

# Activate pyenv
export PYENV_ROOT="/root/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# Activate cbfpy virtual environment
pyenv activate cbfpy-env

exec "$@"