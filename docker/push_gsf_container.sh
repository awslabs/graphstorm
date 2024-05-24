#!/usr/bin/env bash

set -euox pipefail
# Usage help
if [ -b "${1-}" ] && [ "$1" == "--help" ] || [ -b "${1-}" ] && [ "$1" == "-h" ]; then
    echo "Usage: docker/push_gsf_container.sh <image-name> <tag> <region> <account>"
    echo "Optionally provide the image name, tag, region and account number for the ecr repository"
    echo "For example: docker/push_gsf_container.sh graphstorm sm us-west-2 1234567890"
    exit 1
fi

tag="sm" # needs to be updated anytime there's a new version

# TODO: Use proper flags for these arguments instead of relying on position
# Set the image name/repository
if [ -n "${1-}" ]; then
    image="$1"
else
    image='graphstorm'
fi

# Set the image tag/version
if [ -n "${2-}" ]; then
    version="$2"
else
    version=${tag}
fi

# Get the region defined in the current configuration (default to us-west-2 if none defined)
if [ -n "${3-}" ]; then
    region="$3"
else
    region=$(aws configure get region)
    region=${region:-us-west-2}
fi

# Get the account number associated with the current IAM credentials
if [ -n "${4-}" ]; then
    account=$4
else
    account=$(aws sts get-caller-identity --query Account --output text)
fi

suffix="${version}"
latest_suffix="latest"

echo "ecr image: ${image},
version: ${version},
region: ${region},
account: ${account}"

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${suffix}"
latest_tag="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${latest_suffix}"

# If the repository doesn't exist in ECR, create it.
echo "Getting or creating container repository: ${image}"
if ! $(aws ecr describe-repositories --repository-names "${image}" --region ${region} > /dev/null 2>&1); then
    echo "Container repository ${image} does not exist. Creating"
    aws ecr create-repository --repository-name "${image}" --region ${region} > /dev/null
fi

echo "Get the docker login command from ECR and execute it directly"
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

echo "Pushing image ${fullname}"

docker tag ${image}:${suffix} ${fullname}

docker push ${fullname}
