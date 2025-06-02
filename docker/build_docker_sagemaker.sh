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

# process argument 2: docker image device type, default is GPU
if [ -z "$2" ]; then
    DEVICE_TYPE="gpu"
else
    DEVICE_TYPE="$2"
fi

# process argument 3: docker image type, options include training, infer. Default is train
if [ -z "$3" ]; then
    IMAGGE_TYPE="train"
else
    IMAGE_TYPE="$3"
fi

# process argument 4: docker image name, default is graphstorm
if [ -z "$4" ]; then
    IMAGE_NAME="graphstorm"
else
    IMAGE_NAME="$4"
fi

# process argument 5: image's tag name, default is sm
if [ -z "$5" ]; then
    TAG="sm"
else
    TAG="$5"
fi

# Copy scripts and tools codes to the docker folder
# TODO: use pip install later
mkdir -p code/graphstorm
cp -r "${GSF_HOME}/python" code/graphstorm/
cp -r "${GSF_HOME}/sagemaker" code/graphstorm/sagemaker
cp -r "${GSF_HOME}/docker/sagemaker/build_artifacts" build_artifacts

# Log in to ECR to pull Docker image
aws ecr get-login-password --region us-east-1 \
        | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

if [ $IMAGE_TYPE = "train" ] || [ $IMAGE_TYPE = "infer" ]; then
    if [ $IMAGE_TYPE = "train" ]; then
        DOCKER_FILE="${GSF_HOME}/docker/sagemaker/Dockerfile.sm"
    else
        DOCKER_FILE="${GSF_HOME}/docker/sagemaker/Dockerfile-infer.sm"
    fi
else
    echo "Image type can only be \"train\" or \"infer\", but got \""$IMAGE_TYPE"\""
    # remove the temporary code folder
    rm -rf code
    exit 1
fi

# Build OSS docker for SageMaker that an pull ECR docker images
DOCKER_FULLNAME="${IMAGE_NAME}:${TAG}-${IMAGE_TYPE}-${DEVICE_TYPE}"

echo "Build a sagemaker docker image ${DOCKER_FULLNAME}"

if [ $DEVICE_TYPE = "gpu" ] || [ $DEVICE_TYPE = "cpu" ]; then
    # Use Buildkit to avoid pulling both CPU and GPU images
    DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=$DEVICE_TYPE \
        -f $DOCKER_FILE . -t $DOCKER_FULLNAME
else
    echo "Device type can only be \"gpu\" or \"cpu\", but got \""$DEVICE_TYPE"\""
    # remove the temporary code folder
    rm -rf code
    exit 1
fi

# remove the temporary code folder
rm -rf code
rm -rf build_artifacts
