#!/usr/bin/env bash
set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

usage() {
    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [-h] [-x] [--image ...] [--version ...] [--region ...] [--account ...]

Pushes GSProcessing image to ECR.

Available options:

-h, --help          Print this help and exit
-x, --verbose       Print script debug info (set -x)
-e, --environment   Image execution environment. Must be one of 'local' or 'sagemaker'. Required.
-i, --image         Docker image name, default is 'graphstorm-\${environment}'.
-v, --version       Docker version tag, default is the library's current version
-s, --suffix        Suffix for the image tag, can be used to push custom image tags. Default is "".
-r, --region        AWS Region to which we'll push the image. By default will get from aws-cli configuration.
-a, --account       AWS Account ID. By default will get from aws-cli configuration.
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
    IMAGE='graphstorm'
    VERSION=$(grep "$GSF_HOME/python/graphstorm/__init__.py" __version__ | cut -d " " -f 3)
    SUFFIX=""
    LATEST_VERSION=${VERSION}
    REGION=$(aws configure get region) || REGION=""
    REGION=${REGION:-us-east=1}
    ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

    while :; do
        case "${1-}" in
        -h | --help) usage ;;
        -x | --verbose) set -x ;;
        --no-color) NO_COLOR=1 ;;
        -e | --environment)
            EXEC_ENV="${2-}"
            shift
            ;;
        -i | --image)
            IMAGE="${2-}"
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
}

parse_params "${@}"

if [[ ${EXEC_ENV} == "sagemaker" || ${EXEC_ENV} == "local" ]]; then
    : # Do nothing
else
    die "--environment parameter needs to be one of 'sagemaker', or 'local' got ${EXEC_ENV}"
fi

TAG="${VERSION}-${ARCH}${SUFFIX}"
LATEST_TAG="latest-${ARCH}${SUFFIX}"
IMAGE_WITH_ENV="${IMAGE}-${EXEC_ENV}"

msg "Execution parameters: "
msg "- ENVIRONMENT: ${EXEC_ENV}"
msg "- IMAGE: ${IMAGE}"
msg "- TAG: ${TAG}"
msg "- REGION: ${REGION}"
msg "- ACCOUNT: ${ACCOUNT}"

FULLNAME="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_WITH_ENV}:${TAG}"
LATEST_FULLNAME="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_WITH_ENV}:${LATEST_TAG}"

# If the repository doesn't exist in ECR, create it.
echo "Getting or creating container repository: ${IMAGE_WITH_ENV}"
if ! eval aws ecr describe-repositories --repository-names "${IMAGE_WITH_ENV}" --region ${REGION} >/dev/null 2>&1; then
    echo >&2 "WARNING: ECR repository ${IMAGE_WITH_ENV} does not exist in region ${REGION}. Creating..."
    aws ecr create-repository --repository-name "${IMAGE_WITH_ENV}" --region ${REGION} >/dev/null
fi

echo "Logging into ECR with local credentials"
aws ecr get-login-password --region ${REGION} |
    docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

echo "Pushing image to ${FULLNAME}"

docker tag ${IMAGE_WITH_ENV}:${TAG} ${FULLNAME}

docker push ${FULLNAME}

if [ ${VERSION} = ${LATEST_VERSION} ]; then
    docker tag ${IMAGE_WITH_ENV}:${TAG} ${LATEST_FULLNAME}
    docker push ${LATEST_FULLNAME}
fi
