#!/bin/bash
set -eo pipefail

# process argument 1: graphstorm home folder
if [ -z "$1" ]; then
    echo "Please provide the graphstorm home folder that the graphstorm codes are cloned to."
    echo "For example, ./docker/build_docker_sagemaker.sh /graph-storm/ <cpu|gpu> <image_name> <image_tag> "
    exit 1
else
    GSF_HOME="$1"
fi

# process argument 2: docker image type, default is GPU
if [ -z "$2" ]; then
    IMAGE_TYPE="gpu"
else
    IMAGE_TYPE="$2"
fi

# process argument 3: docker image name, default is graphstorm
if [ -z "$3" ]; then
    IMAGE_NAME="graphstorm"
else
    IMAGE_NAME="$3"
fi

# process argument 4: image's tag name, default is sm
if [ -z "$4" ]; then
    TAG="sm"
else
    TAG="$4"
fi

# Copy scripts and tools codes to the docker folder
# TODO: use pip install later
mkdir -p code/graphstorm
cp -r "${GSF_HOME}/python" code/graphstorm/
cp -r "${GSF_HOME}/sagemaker" code/graphstorm/sagemaker
cp -r "${GSF_HOME}/docker/sagemaker/build_artifacts" build_artifacts

# Build OSS docker for EC2 instances that an pull ECR docker images
DOCKER_FULLNAME="${IMAGE_NAME}:${TAG}"

echo "Build a sagemaker docker image ${DOCKER_FULLNAME}"

if [ $IMAGE_TYPE = "gpu" ] || [ $IMAGE_TYPE = "cpu" ]; then
    docker build --build-arg DEVICE=$IMAGE_TYPE -f $GSF_HOME"docker/sagemaker/Dockerfile.sm" . -t $DOCKER_FULLNAME
else
    echo "Image type can only be \"gpu\" or \"cpu\", but get \""$IMAGE_TYPE"\""
    # remove the temporary code folder
    rm -rf code
    exit 1
fi

# remove the temporary code folder
rm -rf code
rm -rf build_artifacts
