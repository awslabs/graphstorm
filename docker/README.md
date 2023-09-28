# How to build Docker images to run in local instances

GraphStorm can be installed as a pip package. However, running GraphStorm in a distributed environment is non-trivial.
Users need to install dependencies and configure distributed Pytorch running environments. For this reason, we
recommend that our users use Docker as the base running environment to use GraphStorm in a distributed environment.

For users who want to create their own GraphStorm Docker images because they want to add additional functions,
e.g. graph data building, you can use the provided scripts to build your own GraphStorm Docker images.

## Prerequisites
-----------------
You need to install Docker in your environment as the [Docker documentation](https://docs.docker.com/get-docker/)
suggests.

For example, in an AWS EC2 instance created with Deep Learning AMI GPU PyTorch 1.13.0, you can run
the following commands to install Docker.
```shell
sudo apt-get update
sudo apt update
sudo apt install Docker.io
```

## Build a Docker image from source
---------------

Once you have the GraphStorm repository cloned, please use the following command to build a Docker image from source:
```shell
cd /path-to-graphstorm/docker/

bash /path-to-graphstorm/docker/build_docker_oss4local.sh /path-to-graphstorm/ docker-name docker-tag
```

There are three arguments of the `build_docker_oss4local.sh`:

1. **path-to-graphstorm**(required), is the absolute path of the "graphstorm" folder, where you
cloned the GraphStorm source code. For example, the path could be "/code/graphstorm".
2. **docker-name**(optional), is the assigned name of the to be built Docker image. Default is
"graphstorm".
3. **docker-tag**(optional), is the assigned tag name of the to be built docker image. Default is
"local".

If Docker requires you to run it as a root user and you don't want to preface all docker commands with sudo, you can check the solution available [here](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

You can use the below command to check if the new image exists.
```shell
docker image ls
```
If the build succeeds, there should be a new Docker image, named `<docker-name>:<docker-tag>`, e.g., "graphstorm:local".
