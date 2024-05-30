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

bash /path-to-graphstorm/docker/build_docker_oss4local.sh /path-to-graphstorm/ image-name image-tag device
```

There are four arguments of the `build_docker_oss4local.sh`:

1. **path-to-graphstorm**(required), is the absolute path of the "graphstorm" folder, where you
cloned the GraphStorm source code. For example, the path could be "/code/graphstorm".
2. **docker-name**(optional), is the assigned name of the to be built Docker image. Default is
"graphstorm".
3. **docker-tag**(optional), is the assigned tag name of the to be built docker image. Default is
"local".
4. **device**(optional), is the intended execution device for the image. Should be one of `cpu` or `gpu`, default is
`gpu`.

If Docker requires you to run it as a root user and you don't want to preface all docker commands with sudo, you can check the solution available [here](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

You can use the below command to check if the new image exists.
```shell
docker image ls
```
If the build succeeds, there should be a new Docker image, named `<image-name>:<image-tag>-<device>`, e.g., "graphstorm:local-gpu".

To push the image to ECR you can use the `push_gsf_container.sh` script.
It takes 4 positional arguments,  `image-name` `image-tag-device`, `region`, and `account`.
For example to push the local GPU image to the us-west-2 on AWS account `1234567890` use:

```bash
bash docker/push_gsf_container.sh graphstorm local-gpu us-west-2 1234567890
```

## Using a customer DGL codebase
---------------
To use a local DGL codebase, you'll need to modify the build script and Dockerfile.local.


You can add the following to the build_docker_oss4local.sh:

```bash
mkdir -p code/dgl
rsync -qr "${GSF_HOME}/../dgl/" code/dgl/ --exclude .venv --exclude dist --exclude ".*/" \
        --exclude "*__pycache__" --exclude "third_party"
```

and in `local/Dockerfile.local` replace the line `RUN cd /root; git clone --branch v${DGL_VERSION} https://github.com/dmlc/dgl.git`
with the following lines:

```Dockerfile
COPY code/dgl /root/dgl
ENV PYTHONPATH="/root/dgl/python/:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/opt/gs-venv/lib/python3.9/site-packages/dgl/:$LD_LIBRARY_PATH"
```