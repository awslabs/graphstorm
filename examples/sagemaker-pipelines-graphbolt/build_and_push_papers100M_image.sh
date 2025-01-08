#!/usr/bin/env bash
set -xEeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
  # script cleanup here
  rm -f ripunzip_2.0.0-1_amd64.deb
}


die() {
    local msg=$1
    local code=${2-1} # default exit status 1
    msg "$msg"
    exit "$code"
}

parse_params() {
    # default values of variables set from params
    ACCOUNT=$(aws sts get-caller-identity --query Account --output text || true)
    REGION=$(aws configure get region || true)
    REGION=${REGION:-"us-east-1"}

    while :; do
        case "${1-}" in
        -h | --help) usage ;;
        -x | --verbose) set -x ;;
        -a | --account)
            ACCOUNT="${2-}"
            shift
            ;;
        -r | --region)
            REGION="${2-}"
            shift
            ;;
        -?*) die "Unknown option: $1" ;;
        *) break ;;
        esac
        shift
    done

    # check required params and arguments
    [[ -z "${ACCOUNT-}" ]] && die "Missing required parameter: -a/--account <aws-account-id>"
    [[ -z "${REGION-}" ]] && die "Missing required parameter: -r/--region <aws-region>"

    return 0
}

parse_params "$@"

IMAGE=papers100m-processor

# Download ripunzip to copy to image
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
