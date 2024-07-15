#!/usr/bin/env bash

set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [-h] [-x] -e sagemaker

Builds the GraphStorm Processing Docker image.

Available options:

-h, --help          Print this help and exit
-x, --verbose       Print script debug info (set -x)
-e, --environment   Image execution environment. Must be one of 'emr', 'emr-serverless' or 'sagemaker'. Required.
-a, --architecture  Image architecture. Must be one of 'x86_64' or 'arm64'. Default is 'x86_64'.
                    Note that only x86_64 architecture is supported for SageMaker.
-t, --target        Docker image target, must be one of 'prod' or 'test'. Default is 'prod'.
-p, --path          Path to graphstorm-processing root directory, default is one level above this script's location.
-i, --image         Docker image name, default is 'graphstorm-processing-\${environment}'.
-v, --version       Docker version tag, default is the library's current version (`poetry version --short`)
-s, --suffix        Suffix for the image tag, can be used to push custom image tags. Default is "".
-b, --build         Docker build directory prefix, default is '/tmp/'.
-m, --hf-model      Provide a Huggingface Model name to be packed into the docker image. Default is "", no model included.

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
  GSP_HOME="${SCRIPT_DIR}/../"
  IMAGE_NAME='graphstorm-processing'
  VERSION=`poetry version --short`
  BUILD_DIR='/tmp'
  TARGET='prod'
  ARCH='x86_64'
  SUFFIX=""
  MODEL=""

  while :; do
    case "${1-}" in
    -h | --help) usage ;;
    -x | --verbose) set -x ;;
    --no-color) NO_COLOR=1 ;;
    -t | --target)
      TARGET="${2-}"
      shift
      ;;
    -e | --environment)
      EXEC_ENV="${2-}"
      shift
      ;;
    -a | --architecture)
      ARCH="${2-}"
      shift
      ;;
    -p | --path)
      GSP_HOME="${2-}"
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
    -m | --hf-model)
      MODEL="${2-}"
      shift
      ;;
    -?*) die "Unknown option: $1" ;;
    *) break ;;
    esac
    shift
  done

  args=("$@")

  # check required params and arguments
  [[ -z "${EXEC_ENV-}" ]] && die "Missing required parameter: -e/--environment [emr|emr-serverless|sagemaker]"

  return 0
}

cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
  # script cleanup here
  if [[ $BUILD_DIR ]]; then
    rm -rf "${BUILD_DIR}/docker/code"
  fi
}

parse_params "$@"

if [[ ${EXEC_ENV} == "emr" || ${EXEC_ENV} == "emr-serverless" || ${EXEC_ENV} == "sagemaker" ]]; then
    :  # Do nothing
else
    die "--environment parameter needs to be one of 'emr', 'emr-serverless' or 'sagemaker', got ${EXEC_ENV}"
fi


if [[ ${TARGET} == "prod" || ${TARGET} == "test" ]]; then
    :  # Do nothing
else
    die "--target parameter needs to be one of 'prod' or 'test', got ${TARGET}"
fi

if [[ ${ARCH} == "x86_64" || ${ARCH} == "arm64" ]]; then
    :  # Do nothing
else
    die "--architecture parameter needs to be one of 'arm64' or 'x86_64', got ${ARCH}"
fi

if [[ ${EXEC_ENV} == "sagemaker" && ${ARCH} == "arm64" ]]; then
    die "arm64 architecture is not supported for SageMaker"
fi

# TODO: Ensure that the version requested has a corresponding directory

# script logic here
msg "Execution parameters:"
msg "- ENVIRONMENT: ${EXEC_ENV}"
msg "- ARCHITECTURE: ${ARCH}"
msg "- TARGET: ${TARGET}"
msg "- GSP_HOME: ${GSP_HOME}"
msg "- IMAGE_NAME: ${IMAGE_NAME}"
msg "- VERSION: ${VERSION}"
msg "- SUFFIX: ${SUFFIX}"
msg "- MODEL: ${MODEL}"

# Prepare Docker build directory
rm -rf "${BUILD_DIR}/docker/code"
mkdir -p "${BUILD_DIR}/docker/code"

if [[ ${TARGET} == "prod" ]]; then
    # Build the graphstorm-processing library and copy the wheels file
    poetry build -C ${GSP_HOME} --format wheel
    cp ${GSP_HOME}/dist/graphstorm_processing-${VERSION}-py3-none-any.whl \
        "${BUILD_DIR}/docker/code"
else
    # Copy library source code along with test files
    rsync -r ${GSP_HOME} "${BUILD_DIR}/docker/code/graphstorm-processing/" --exclude .venv --exclude dist \
        --exclude "*__pycache__" --exclude "*.pytest_cache" --exclude "*.mypy_cache"
    cp ${GSP_HOME}/../graphstorm_job.sh "${BUILD_DIR}/docker/code/"
fi

# Copy Docker entry point to build folder
cp ${GSP_HOME}/docker-entry.sh "${BUILD_DIR}/docker/code/"

# Export Poetry requirements to requirements.txt file
poetry export -f requirements.txt --output "${BUILD_DIR}/docker/requirements.txt"

# Set image name
DOCKER_FULLNAME="${IMAGE_NAME}-${EXEC_ENV}:${VERSION}-${ARCH}${SUFFIX}"

# Login to ECR to be able to pull source SageMaker image
if [[ ${EXEC_ENV} == "sagemaker" ]]; then
    aws ecr get-login-password --region us-west-2 \
        | docker login --username AWS --password-stdin 153931337802.dkr.ecr.us-west-2.amazonaws.com
else
    aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
fi

echo "Build a Docker image ${DOCKER_FULLNAME}"
DOCKER_BUILDKIT=1 docker build --platform "linux/${ARCH}" -f "${GSP_HOME}/docker/${VERSION}/${EXEC_ENV}/Dockerfile.cpu" \
    "${BUILD_DIR}/docker/" -t $DOCKER_FULLNAME --target ${TARGET} --build-arg ARCH=${ARCH} --build-arg MODEL=${MODEL}
