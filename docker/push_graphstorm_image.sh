#!/usr/bin/env bash
set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

usage() {
    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [-h] [-x] -e/--environment [sagemaker|local] [--region ...] [--account ...]

Pushes GSProcessing image to ECR.

Available options:

-h, --help          Print this help and exit
-x, --verbose       Print script debug info (set -x)
-e, --environment   Image execution environment. Must be one of 'local' or 'sagemaker'. Required.
-d, --device        Device type. Must be one of 'gpu' or 'cpu'. Default is 'gpu'.
-i, --image         Docker image name, default is 'graphstorm'.
-s, --suffix        Suffix for the image tag, can be used to push custom image tags. Default tag is "<environment>-<device>".
-r, --region        AWS Region to which we'll push the image. By default will get from aws-cli configuration.
-a, --account       AWS Account ID. By default will get from aws-cli configuration.

Example:

    bash $(basename "${BASH_SOURCE[0]}") -e sagemaker --device cpu --account 123456789012 --region us-east-1
    # Will push an image to '123456789012.dkr.ecr.us-east-1.amazonaws.com/graphstorm:sagemaker-cpu'

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
    IMAGE_NAME='graphstorm'
    SUFFIX=""
    REGION=$(aws configure get region) || REGION=""
    REGION=${REGION:-us-east=1}
    ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

    while :; do
        case "${1-}" in
        -h | --help) usage ;;
        -x | --verbose) set -x ;;
        -e | --environment)
            EXEC_ENV="${2-}"
            shift
            ;;
        -i | --image)
            IMAGE_NAME="${2-}"
            shift
            ;;
        -d | --device)
            DEVICE_TYPE="${2-}"
            shift
            ;;
        -s | --suffix)
            SUFFIX="${2-}"
            shift
            ;;
        -r | --region)
            REGION="${2-}"
            shift
            ;;
        -a | --account)
            ACCOUNT="${2-}"
            shift
            ;;
        -?*) die "Unknown option: $1" ;;
        *) break ;;
        esac
        shift
    done

    [[ -z "${EXEC_ENV-}" ]] && die "Missing required parameter: -e/--environment [local|sagemaker]"

    return 0
}

cleanup() {
    trap - SIGINT SIGTERM ERR EXIT
    # script cleanup here
    rm -f /tmp/ecr_error
}

parse_params "${@}"

if [[ ${EXEC_ENV} == "sagemaker" || ${EXEC_ENV} == "local" ]]; then
    : # Do nothing
else
    die "--environment parameter needs to be one of 'sagemaker', or 'local' got ${EXEC_ENV}"
fi

TAG="${EXEC_ENV}-${DEVICE_TYPE}${SUFFIX}"
IMAGE="${IMAGE_NAME}"

msg "Execution parameters: "
msg "- ENVIRONMENT: ${EXEC_ENV}"
msg "- DEVICE TYPE: ${DEVICE_TYPE}"
msg "- IMAGE: ${IMAGE}"
msg "- TAG: ${TAG}"
msg "- REGION: ${REGION}"
msg "- ACCOUNT: ${ACCOUNT}"

FULLNAME="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE}:${TAG}"

# If the repository doesn't exist in ECR, create it.
echo "Getting or creating container repository: ${IMAGE}"
if ! eval aws ecr describe-repositories --repository-names "${IMAGE}" --region ${REGION} >/dev/null 2>&1; then
    msg "WARNING: ECR repository ${IMAGE} does not exist in region ${REGION}. Attempting to create..."

    if ! aws ecr create-repository --repository-name "${IMAGE}" --region ${REGION} 2>/tmp/ecr_error; then
        error_msg=$(cat /tmp/ecr_error)
        if echo "$error_msg" | grep -q "AccessDeniedException"; then
            msg "ERROR: You don't have sufficient permissions to create ECR repository"
            msg "Required permission: ecr:CreateRepository"
            exit 1
        else
            msg "ERROR: Failed to create ECR repository: ${error_msg}"
            exit 1
        fi
    fi
    msg "Successfully created ECR repository ${IMAGE}"
fi

msg "Logging into ECR with local credentials"
aws ecr get-login-password --region ${REGION} |
    docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

msg "Pushing image to ${FULLNAME}"

docker tag "${IMAGE}:${TAG}" "${FULLNAME}"

docker push "${FULLNAME}"
