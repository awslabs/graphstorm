#!/usr/bin/env bash

set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)

usage() {
    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [-h] [-x] -e sagemaker

Builds the GraphStorm training/inference Docker images.

Available options:

-h, --help          Print this help and exit
-x, --verbose       Print script debug info (set -x)
-e, --environment   Image execution environment. Must be one of 'local' or 'sagemaker'. Required.
-p, --path          Path to graphstorm root directory, default is one level above this script's location.
-i, --image         Docker image name, default is 'graphstorm-\${environment}'.
-v, --version       Docker version tag, default is the library's current version
-s, --suffix        Suffix for the image tag, can be used to push custom image tags. Default is "".
-b, --build         Docker build directory prefix, default is '/tmp/'.
--use-parmetis      Include dependencies to be able to run ParMETIS distributed partitioning.

Example:

    bash $(basename "${BASH_SOURCE[0]}") -e sagemaker -v 0.4.0 --device cpu
    # Will build an image tagged as 'graphstorm-sagemaker:0.4.0-cpu'

EOF
    exit
}

msg() {
    echo >&2 -e "${1-}"
}

die() {
    local msg=$1
    local code=${2-1} # default exit status 1
    msg "$msg"
    exit "$code"
}

parse_params() {
    # default values of variables set from params
    GSF_HOME="${SCRIPT_DIR}/../"
    IMAGE_NAME='graphstorm'
    VERSION=$(grep "$GSF_HOME/python/graphstorm/__init__.py" __version__ | cut -d " " -f 3)
    USE_PARMETIS=false
    BUILD_DIR='/tmp/graphstorm-build'
    SUFFIX=""

    while :; do
        case "${1-}" in
        -h | --help) usage ;;
        -x | --verbose) set -x ;;
        -e | --environment)
            EXEC_ENV="${2-}"
            shift
            ;;
        -p | --path)
            GSF_HOME="${2-}"
            shift
            ;;
        -b | --build)
            BUILD_DIR="${2-}"
            shift
            ;;
        -i | --image)
            IMAGE_NAME="${2-}"
            shift
            ;;
        -v | --version)
            VERSION="${2-}"
            shift
            ;;
        -s | --suffix)
            SUFFIX="${2-}"
            shift
            ;;
        --use-parmetis)
            USE_PARMETIS=true
            shift
            ;;
        -?*) die "Unknown option: $1" ;;
        *) break ;;
        esac
        shift
    done

    # check required params and arguments
    [[ -z "${EXEC_ENV-}" ]] && die "Missing required parameter: -e/--environment [local|sagemaker]"

    return 0
}

cleanup() {
    trap - SIGINT SIGTERM ERR EXIT
    # script cleanup here
    if [[ ${BUILD_DIR} ]]; then
        rm -rf "${BUILD_DIR}/docker/code"
    fi
}

parse_params "$@"

if [[ ${EXEC_ENV} == "local" || ${EXEC_ENV} == "sagemaker" ]]; then
    : # Do nothing
else
    die "--environment parameter needs to be one of 'local', '' or 'sagemaker', got ${EXEC_ENV}"
fi

# Print build parameters
msg "Execution parameters:"
msg "- EXECUTION ENVIRONMENT: ${EXEC_ENV}"
msg "- GSF_HOME: ${GSF_HOME}"
msg "- IMAGE_NAME: ${IMAGE_NAME}"
msg "- VERSION: ${VERSION}"
msg "- SUFFIX: ${SUFFIX}"
msg "- USE_PARMETIS: ${USE_PARMETIS}"

# Prepare Docker build directory
rm -rf "${BUILD_DIR}/docker/code"
mkdir -p "${BUILD_DIR}/docker/code"

# Set image name
DOCKER_FULLNAME="${IMAGE_NAME}-${EXEC_ENV}:${VERSION}-${ARCH}${SUFFIX}"

# Login to ECR to be able to pull source SageMaker or public.ecr.aws image
msg "Authenticating to public ECR registry"
if [[ ${EXEC_ENV} == "sagemaker" ]]; then
    aws ecr get-login-password --region us-east-1 \
        | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
else
    # Using local image, login to public ECR
    aws ecr-public get-login-password --region us-east-1 |
        docker login --username AWS --password-stdin public.ecr.aws
fi

# Copy scripts and tools codes to the docker folder
mkdir -p $GSF_HOME"/docker/code"

cp -r $GSF_HOME"/python" $GSF_HOME"/docker/code/python"
cp -r $GSF_HOME"/examples" $GSF_HOME"/docker/code/examples"
cp -r $GSF_HOME"/inference_scripts" $GSF_HOME"/docker/code/inference_scripts"
cp -r $GSF_HOME"/tools" $GSF_HOME"/docker/code/tools"
cp -r $GSF_HOME"/training_scripts" $GSF_HOME"/docker/code/training_scripts"

DOCKER_FULLNAME="${IMAGE_NAME}:${TAG}-${DEVICE_TYPE}"


if [[ $EXEC_ENV = "local" ]]; then
    cp $SCRIPT_DIR"/local/fetch_and_run.sh" $GSF_HOME"/docker/code/"
    DOCKERFILE="${GSF_HOME}/docker/local/Dockerfile.local"

    if [[ $DEVICE_TYPE = "gpu" ]]; then
        SOURCE_IMAGE="nvidia/cuda:12.1.1-runtime-ubuntu22.04"
    else
        SOURCE_IMAGE="public.ecr.aws/ubuntu/ubuntu:22.04_stable"
    fi
elif [[ $EXEC_ENV = "sagemaker" ]]; then
    DOCKERFILE="${GSF_HOME}/docker/sagemaker/Dockerfile.sm"
    if [[ $DEVICE_TYPE = "gpu" ]]; then
        SOURCE_IMAGE="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker"
    elif [[ $DEVICE_TYPE = "cpu" ]]; then
        SOURCE_IMAGE="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.3.0-cpu-py311-ubuntu20.04-sagemaker"
    fi
fi

# Use Buildkit to avoid pulling both CPU and GPU images
echo "Building Docker image: ${DOCKER_FULLNAME}"
DOCKER_BUILDKIT=1 docker build \
    --build-arg DEVICE=$DEVICE_TYPE \
    --build-arg SOURCE=${SOURCE_IMAGE} \
    --build-arg USE_PARMETIS=${USE_PARMETIS} \
    -f "$DOCKERFILE" . -t "$DOCKER_FULLNAME"
