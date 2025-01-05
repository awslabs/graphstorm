#!/usr/bin/env bash
set -xEeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
  # script cleanup here
  rm -f ripunzip_2.0.0-1_amd64.deb
}


ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
REGION=${REGION:-us-east-1}
IMAGE=papers100m-processor

curl -L -O https://github.com/google/ripunzip/releases/download/v2.0.0/ripunzip_2.0.0-1_amd64.deb

# Auth to AWS public ECR gallery
aws ecr-public get-login-password --region $REGION | docker login --username AWS --password-stdin public.ecr.aws

# Build and tag image
docker build -f Dockerfile.processing -t $IMAGE .


# Create repository if it doesn't exist
echo "Getting or creating container repository: $IMAGE"
if ! $(aws ecr describe-repositories --repository-names $IMAGE --region ${REGION} > /dev/null 2>&1); then
    echo >&2 "WARNING: ECR repository $IMAGE does not exist in region ${REGION}. Creating..."
    aws ecr create-repository --repository-name $IMAGE --region ${REGION} > /dev/null
fi

# Auth to private ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

# Tag and push the image
docker tag $IMAGE:latest $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$IMAGE:latest

docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$IMAGE:latest
