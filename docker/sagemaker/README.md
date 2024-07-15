# How to build docker images to run GraphStorm on SageMaker
GraphStorm can run on Amazon Sagemaker. This doc shows how to build a SageMaker compatible GraphStorm docker image.

## Prerequisites
-----------------
You need to install Docker in your environment as the [Docker documentation](https://docs.docker.com/get-docker/) suggests.


For example, in an AWS EC2 instance created with Deep Learning AMI GPU PyTorch 1.13.0, you can run
the following commands to install Docker.
```shell
sudo apt-get update
sudo apt update
sudo apt install Docker.io
```

You need to have the access to pull [SageMaker Framework Contaienrs](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#sagemaker-framework-containers-sm-support-only).
Please follow [AWS Deep Learning Containers](https://github.com/aws/deep-learning-containers) guideline to get access to the image.

You also need to have the access to [Amazon SageMaker service](https://aws.amazon.com/pm/sagemaker).
To do this you can follow the [SageMaker Getting set-up guide](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html).

## Build a Docker image from source
---------------
Please use the following command to build a Docker image from source:
```bash
mkdir ~/build-docker/
cd ~/build-docker/

bash /path-to-graphstorm/docker/build_docker_sagemaker.sh /path-to-graphstorm/
```

You can change your Docker image type to determine if use GPU or CPU by:
```
bash /path-to-graphstorm/docker/build_docker_sagemaker.sh /path-to-graphstorm/ <gpu|cpu>
```
The default setting is to build a docker image for using GPUs.

You can rename your Docker image by:
```
bash /path-to-graphstorm/docker/build_docker_sagemaker.sh /path-to-graphstorm/ <DOCKER_NAME> <DOCKER_TAG>
```
The default setting of the name and tag of images is `graphstorm` and `sm`.