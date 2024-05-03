#!/bin/bash
set -eox pipefail

# process argument 1: graphstorm home folder
if [ -z "$1" ]; then
    echo "Please provide a path to the root directory of the GraphStorm repository."
    echo "For example, ./build_docker_parmetis.sh ../ graphstorm-parmetis local"
    exit 1
else
    GSF_HOME="$1"
fi

# process argument 2: docker image name, default is graphstorm
if [ -z "$2" ]; then
    IMAGE_NAME="graphstorm-parmetis"
else
    IMAGE_NAME="$2"
fi

# process argument 3: image's tag name, default is local
if [ -z "$3" ]; then
    TAG="local"
else
    TAG="$3"
fi

# Copy scripts and tools codes to the docker folder
mkdir -p $GSF_HOME"/docker/code"
cp -r $GSF_HOME"/python" $GSF_HOME"/docker/code/python"
cp -r $GSF_HOME"/examples" $GSF_HOME"/docker/code/examples"
cp -r $GSF_HOME"/inference_scripts" $GSF_HOME"/docker/code/inference_scripts"
cp -r $GSF_HOME"/tools" $GSF_HOME"/docker/code/tools"
cp -r $GSF_HOME"/training_scripts" $GSF_HOME"/docker/code/training_scripts"

aws ecr-public get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin public.ecr.aws

# Build OSS docker for EC2 instances that an pull ECR docker images
DOCKER_FULLNAME="${IMAGE_NAME}:${TAG}"

echo "Build a local docker image ${DOCKER_FULLNAME}"

SOURCE_IMAGE="public.ecr.aws/ubuntu/ubuntu:20.04_stable"

# Use Buildkit to avoid pulling both CPU and GPU images
DOCKER_BUILDKIT=1 docker build \
    --build-arg SOURCE=${SOURCE_IMAGE} \
    -f "${GSF_HOME}/docker/parmetis/Dockerfile.parmetis" . -t $DOCKER_FULLNAME

# remove the temporary code folder
rm -rf $GSF_HOME"/docker/code"
