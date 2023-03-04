# How to build Docker images to run in local instances

GraphStorm can be installed as a pip package. However, 
running GraphStorm in a distributed environment is non-trivial. Users need to install dependencies and configure
distributed Pytorch running environments. For this reason, we recommend that our users use Docker as 
the base running environment to use GraphStorm in a distributed environment.

GraphStorm provides Amazon compatible Docker images for users to leverage GraphStorm in 
a more cost-effective and secure way. For users who want to create their own GraphStorm Docker images to add
additional functions, e.g. graph data building, you can use the provided 
scripts to build your own GraphStorm Docker images.

There are two types of Docker building scripts:

1. Build a Docker image with GraphStorm's pip package and add GraphStorm's support tools and scripts.
2. Build a Docker image from the GraphStorm's source code.

## Prerequisites
-----------------
You need to install Docker in your environment as the [Docker documentation](https://docs.Docker.com/get-Docker/) 
suggests.

For example, in an AWS EC2 instance created with Deep Learning AMI GPU PyTorch 1.12.0, you can run
the following commands to install Docker.
```shell
sudo apt-get update
sudo apt update
sudo apt install Docker.io
```

## Build a Docker image with pip package
---------------
This case is good for users who want to use stable GraphStorm releases but also want to 
add additional GraphStorm support functions such as customized graph processing pipeline or 
customized GNN models.

This case is also good for building Docker images with stable GraphStorm releases and upload
to the Amazon ECR repository for AWS customers.

Please use the following command to build a Docker image with pip package
```shell
bash /path-to-graph-storm/docker/build_docker_oss4ecr.sh /path-to-graph-storm/ pip-package-version docker-tag
```
There are three arguments of the `build_docker_oss4ecr.sh`:

1. **path-to-graph-storm**(required), is the absolute path of the "graph-storm" folder, where you 
clone the GraphStorm source code. For example, the path could be "/code/graph-storm".
2. **pip-package-version**(required), is the GraphStorm pip installation version to build this 
Docker image, e.g., "0.0.1+dda85537".
3. **docker-tag**(optional), is the assigned tag name of the to be built Docker image. Default is 
"ec2_v1".

You can use the below command to check if the new image exists. 
```shell
docker image ls
```
If the build succeeds, there should be a new Docker image, named `graphstorm:<docker-tag>`, e.g. "graphstorm:ec2_v1".

**Note**: If you are using an AWS EC2 instance, have valid Amazon ECR repository, and configure the
 EC2 instance with the access key ID and security access key, the script can automatically push the built Docker image into your ECR 
`us-east-1` repository. In such case, the image name is `<account_id>.dkr.ecr.us-east-1.amazonaws.com/graphstorm_oss:<docker-tag>`.

## Build a Docker image from source
---------------
This case is good for users who want to modify the core code of GraphStorm and apply the 
modification to their problem but do not want to spend time on environment configurations.

Please use the following command to build a Docker image from source:
```shell
bash /path-to-graph-storm/docker/build_docker_oss4local.sh /path-to-graph-storm/ docker-name docker-tag
```

There are three arguments of the `build_docker_oss4local.sh`:

1. **path-to-graph-storm**(required), is the absolute path of the "graph-storm" folder, where you 
clone the GraphStorm source code. For example, the path could be "/code/graph-storm".
2. **docker-name**(optional), is the assigned name of the to be built Docker image. Default is 
"graphstorm".
3. **docker-tag**(optional), is the assigned tag name of the to be built docker image. Default is 
"local".

You can use the below command to check if the new image exists. 
```shell
docker image ls
```
If the build succeeds, there should be a new Docker image, named `<docker-name>:<docker-tag>`, e.g., "graphstorm:local".
