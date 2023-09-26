#!/bin/bash

# process argument 1: graphstorm home folder
if [ -z "$1" ]; then
    echo "Please provide the graphstorm home folder that the graphstorm codes are cloned to."
    echo "For example, ./build_docker_wholegraph.sh /graph-storm/"
    exit 1
else
    GSF_HOME="$1"
fi

# process argument 2: docker image name, default is graphstorm
if [ -z "$2" ]; then
    IMAGE_NAME="graphstorm-wholegraph"
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
cp -r $GSF_HOME"/inference_scripts" $GSF_HOME"/docker/code/inference_scripts"
cp -r $GSF_HOME"/tools" $GSF_HOME"/docker/code/tools"
cp -r $GSF_HOME"/training_scripts" $GSF_HOME"/docker/code/training_scripts"

# Build OSS docker for EC2 instances that an pull ECR docker images
DOCKER_FULLNAME="${IMAGE_NAME}:${TAG}"

echo "Build a local docker image ${DOCKER_FULLNAME}"
docker build --no-cache -f $GSF_HOME"/docker/wholegraph/Dockerfile" . -t $DOCKER_FULLNAME

# remove the temporary code folder
rm -rf $GSF_HOME"/docker/code"
