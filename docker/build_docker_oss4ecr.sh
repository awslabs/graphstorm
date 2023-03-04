#!/bin/bash

# process argument 1: graphstorm home folder 
if [ -z "$1" ]; then
    echo "Please provide the graphstorm home folder that the graphstorm codes are cloned to."
    echo "For example, ./build_docker_oss4local.sh /graph-storm/"
    exit 1
else
    GSF_HOME="$1"
fi

# process argument 2: pip install version. Default is the current branch's first 8 letters of the lastest commit hash 
if [ -z "$2" ]; then
    # Pull the graphstorm pip installation package from the S3 bucket
    # TOD: after OSS, use pip install directly.
    COMMIT_HASH=`git rev-parse HEAD`
    echo "The current commit hash: $COMMIT_HASH"
    VERSION="0.0.1+"${COMMIT_HASH:0:8}
    echo ""
else
    VERSION="$2"
fi

# TODO: 
#    1. need to install awscli tools in the CD instances.
#    2. need to configure CD instances to access the specific S3 bucket.
GSF_WHEEL=$GSF_HOME"docker/graphstorm-${VERSION}-py3-none-any.whl"

if test -f "$GSF_WHEEL"; then
    echo "Use the stored file: $GSF_WHEEL"
else
    echo "Wheel file $GSF_WHEEL does NOT exist. Will copy from the S3 bucket."
    # need to configure AKSK for S3 operation
    aws s3 cp "s3://graphstorm-artifacts/graphstorm-${VERSION}-py3-none-any.whl" ${GSF_HOME}"docker/graphstorm-${VERSION}-py3-none-any.whl"
fi

# process argument 3: image's tag name, default is ec2_v1
if [ -z "$3" ]; then
    TAG="ec2_v1"
else
    TAG="$3"
fi

# Copy scripts and tools codes to the docker folder 
mkdir -p $GSF_HOME"docker/code"
cp -r $GSF_HOME"examples" $GSF_HOME"docker/code/examples"
cp -r $GSF_HOME"inference_scripts" $GSF_HOME"docker/code/inference_scripts"
cp -r $GSF_HOME"tools" $GSF_HOME"docker/code/tools"
cp -r $GSF_HOME"training_scripts" $GSF_HOME"docker/code/training_scripts"

# Build OSS docker for EC2 instances that an pull ECR docker images
ACCOUNT_ID=`aws sts get-caller-identity --query Account --output text`
REGION="us-east-1"

if [ -z $ACCOUNT_ID ]; then
    DOCKER_FULLNAME="graphstorm_oss:${TAG}"
else
    DOCKER_FULLNAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/graphstorm_oss:${TAG}"
fi

echo "Build a local docker image ${DOCKER_FULLNAME}"
docker build -f $GSF_HOME"docker/Dockerfile.pip" . -t $DOCKER_FULLNAME

# push build docker image to ECR
# need to configure AKSK for ECR operation
if [ -z $ACCOUNT_ID ]; then
    echo "Docker image ${DOCKER_FULLNAME} is built ..."
else
    echo "Get the docker login command from ECR and execute it directly"
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
    echo "Push docker image to Amazon ECR repository: ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
    docker push $DOCKER_FULLNAME
fi 

# remove the temporary code folder
rm -rf $GSF_HOME"docker/code"
rm ${GSF_HOME}"docker/graphstorm-${VERSION}-py3-none-any.whl"
