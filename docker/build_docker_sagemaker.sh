#!/bin/bash
set -Eeuo pipefail

# process argument 1: graphstorm home folder
if [ -z "$1" ]; then
    echo "Please provide the graphstorm home folder that the graphstorm codes are cloned to."
    echo "For example, ./docker/build_docker_sagemaker.sh /graph-storm/ <cpu|gpu> <image_name> <image_tag> "
    exit 1
else
    GSF_HOME="$1"
fi

# # process argument 2: docker image type, default is GPU
IMAGE_TYPE="${2:-"gpu"}"
# # process argument 3: docker image name, default is graphstorm
IMAGE_NAME="${3:-"graphstorm"}"
# # process argument 4: image's tag name, default is sm
TAG="${4:-"sm"}"

# Copy scripts and tools codes to the docker folder
# TODO: use pip install later
mkdir -p code/graphstorm
cp -r "${GSF_HOME}/python" code/graphstorm/
cp -r "${GSF_HOME}/sagemaker" code/graphstorm/sagemaker
cp -r "${GSF_HOME}/docker/sagemaker/build_artifacts" build_artifacts

# Build OSS docker for EC2 instances that an pull ECR docker images
DOCKER_FULLNAME="${IMAGE_NAME}:${TAG}"

echo "Build a sagemaker docker image ${DOCKER_FULLNAME}"

# Log in to ECR to pull Docker image
aws ecr get-login-password --region us-east-1 \
        | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

if [ $IMAGE_TYPE = "gpu" ] || [ $IMAGE_TYPE = "cpu" ]; then
    # Use Buildkit to avoid pulling both CPU and GPU images
    DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=$IMAGE_TYPE \
        -f "${GSF_HOME}/docker/sagemaker/Dockerfile.sm" . -t $DOCKER_FULLNAME
else
    echo "Image type can only be \"gpu\" or \"cpu\", but got \""$IMAGE_TYPE"\""
    # remove the temporary code folder
    rm -rf code
    exit 1
fi

# remove the temporary code folder
rm -rf code
rm -rf build_artifacts
