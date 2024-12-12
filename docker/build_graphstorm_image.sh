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
-d, --device        Device type, must be one of 'cpu' or 'gpu'. Default is 'gpu'.
-p, --path          Path to graphstorm root directory, default is one level above this script's location.
-i, --image         Docker image name, default is 'graphstorm'.
-s, --suffix        Suffix for the image tag, can be used to push custom image tags. Default is "<environment>-<device>".
-b, --build         Docker build directory prefix, default is '/tmp/graphstorm-build/docker'.

Example:

    bash $(basename "${BASH_SOURCE[0]}") -e sagemaker --device cpu
    # Will build an image tagged as 'graphstorm:sagemaker-cpu'

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
    DEVICE_TYPE="gpu"
    GSF_HOME="${SCRIPT_DIR}/../"
    IMAGE_NAME='graphstorm'
    BUILD_DIR='/tmp/graphstorm-build/docker'
    SUFFIX=""

    while :; do
        case "${1-}" in
        -h | --help) usage ;;
        -x | --verbose) set -x ;;
        -e | --environment)
            EXEC_ENV="${2-}"
            shift
            ;;
        -d | --device)
            DEVICE_TYPE="${2-}"
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
        -s | --suffix)
            SUFFIX="${2-}"
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
    die "--environment parameter needs to be one of 'local' or 'sagemaker', got ${EXEC_ENV}"
fi

# Print build parameters
msg "Execution parameters:"
msg "- EXECUTION ENVIRONMENT: ${EXEC_ENV}"
msg "- DEVICE_TYPE: ${DEVICE_TYPE}"
msg "- GSF_HOME: ${GSF_HOME}"
msg "- IMAGE_NAME: ${IMAGE_NAME}"
msg "- SUFFIX: ${SUFFIX}"

# Prepare Docker build directory
if [[ -d ${BUILD_DIR} ]]; then
        rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"

# Authenticate to ECR to be able to pull source SageMaker or public.ecr.aws image
msg "Authenticating to public ECR registry"
if [[ ${EXEC_ENV} == "sagemaker" ]]; then
    # Pulling SageMaker image, login to public SageMaker ECR registry
    aws ecr get-login-password --region us-east-1 |
        docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
else
    # Pulling local image, login to Amazon ECR Public Gallery
    aws ecr-public get-login-password --region us-east-1 |
        docker login --username AWS --password-stdin public.ecr.aws
fi

# Prepare Docker build directory
CODE_DIR="${BUILD_DIR}/code"
mkdir -p "${CODE_DIR}"
# TODO: After deprecating the old build scripts, the code copying commands
# can be merged for both local and sagemaker environments, but will
# need Dockerfile changes to support both.


# Set image name
DOCKER_FULLNAME="${IMAGE_NAME}:${EXEC_ENV}-${DEVICE_TYPE}${SUFFIX}"

if [[ $EXEC_ENV = "local" ]]; then

    cp "$SCRIPT_DIR/local/fetch_and_run.sh" "$CODE_DIR"
    cp -r "$GSF_HOME/python" "${CODE_DIR}/python"
    cp -r "$GSF_HOME/examples" "${CODE_DIR}/examples"
    cp -r "$GSF_HOME/inference_scripts" "${CODE_DIR}/inference_scripts"
    cp -r "$GSF_HOME/tools" "${CODE_DIR}/tools"
    cp -r "$GSF_HOME/training_scripts" "${CODE_DIR}/training_scripts"

    DOCKERFILE="${GSF_HOME}/docker/local/Dockerfile.local"

    if [[ $DEVICE_TYPE = "gpu" ]]; then
        SOURCE_IMAGE="nvidia/cuda:12.1.1-runtime-ubuntu22.04"
    else
        SOURCE_IMAGE="public.ecr.aws/ubuntu/ubuntu:22.04_stable"
    fi

elif [[ $EXEC_ENV = "sagemaker" ]]; then
    DOCKERFILE="${GSF_HOME}/docker/sagemaker/Dockerfile.sm"
    rsync -a --exclude="*.pyc" --exclude="*.pyo" --exclude="*.pyd" \
        "${GSF_HOME}/python" "$CODE_DIR/graphstorm/"
    cp -r "${GSF_HOME}/sagemaker" "$CODE_DIR/graphstorm/sagemaker"
    cp -r "${GSF_HOME}/docker/sagemaker/build_artifacts" "$BUILD_DIR"

    if [[ $DEVICE_TYPE = "gpu" ]]; then
        SOURCE_IMAGE="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker"
    elif [[ $DEVICE_TYPE = "cpu" ]]; then
        SOURCE_IMAGE="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.3.0-cpu-py311-ubuntu20.04-sagemaker"
    fi
fi

# Use Buildkit to avoid pulling both CPU and GPU images
echo "Building Docker image: ${DOCKER_FULLNAME}"
DOCKER_BUILDKIT=1 docker build \
    --build-arg DEVICE="$DEVICE_TYPE" \
    --build-arg SOURCE="${SOURCE_IMAGE}" \
    -f "$DOCKERFILE" "${BUILD_DIR}" -t "$DOCKER_FULLNAME"
