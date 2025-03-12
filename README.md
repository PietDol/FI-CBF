# FI-CBF
Documentation of feedback-integrated Control Barrier Functions (FI-CBF).

### Installation
A docker container is created to maintain compatibility between the packages.
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

To run a container that is already runned:
```bash
docker run -it fi-cbf bash
```