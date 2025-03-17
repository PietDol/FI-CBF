# Virtual environment documentation
This file contains the documentation about the virtual environment installation. We use a virual environment to make sure that we keep track of all the dependancies. Finally, we will create a dockerfile (or container) which can be used for deployment. 

### Installation ROS2
Follow [these](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) steps to install ROS2. Install the desktop install version of ROS2.

### Installation steps for CBFpy
1. Create the pyenv environment. Follow the steps in [this](https://danielpmorton.github.io/cbfpy/pyenv/) link from CBFpy. We will create a virtual environment called cbfpy.
2. If you already create the environment, enter the virtual environment with the following command: 
    ```bash
    pyenv shell cbfpy
    ```
3. Clone the [CBFpy](https://github.com/danielpmorton/cbfpy/tree/main) repo:
    ```bash
    git clone https://github.com/danielpmorton/cbfpy
    ```
4. And install everything (install it in editable mode so that changes to the packages directly works):
    ```bash
    cd cbfpy
    pip install -e ".[examples]"
    ```
The environment is set up correctly. Everytime you want to use the environment you can use the command from step 2.

### Useful commands
To source the ROS2:
```bash
source /opt/ros/humble/setup.bash
```