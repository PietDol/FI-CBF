# FI-CBF
Documentation of feedback-integrated Control Barrier Functions (FI-CBF).

### Installation docker (not done)
A docker container is created for deployment but it is not FINISHED. For debugging and prototyping we will create a virtual environment with ROS2.
1. Create docker image:
```bash
docker build -t fi-cbf .
```
2. Run the container:
```bash
docker run -it --name fi-cbf-container fi-cbf
```
3. Check for running containers:
```bash
docker ps
```

Get list of all build containers:
```bash
docker ps -a
```

To run a container that is already runned (after this command you can also enter the container in VS Code if Dev Containers extension is added to VS Code):
```bash
docker start fi-cbf-container
```

To stop a container:
```bash
docker stop fi-cbf-container
```

To go inside the terminal of the container:
```bash
docker exec -it fi-cbf-container bash
```

To source ROS2 manually inside the container (if it is correct this happens automatically):
```bash
source /opt/ros/humble/setup.bash
```