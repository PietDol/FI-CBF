#!/bin/bash
set -e

# Source ROS 2 and workspace
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source /ros_ws/install/setup.bash || true" >> ~/.bashrc
source ~/.bashrc

# Explicitly initialize pyenv
export PYENV_ROOT="/root/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# Manually reload pyenv to ensure virtualenvs are recognized
export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Activate cbfpy virtual environment
pyenv exec python --version  # Debugging: Check if pyenv is working
pyenv versions  # Debugging: Check if the virtual environment exists
pyenv activate cbfpy-env  # Activate the virtual environment

exec "$@"