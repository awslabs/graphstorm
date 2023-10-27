#!/usr/bin/env bash

set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [-h] [-x] -t prod

Script description here.

Available options:

-h, --help      Print this help and exit
-x, --verbose   Print script debug info (set -x)
-t, --target    Docker image target, must be one of 'prod' or 'test'. Default is 'test'.
-p, --path      Path to graphstorm-processing directory, default is one level above this script.
-i, --image     Docker image name, default is 'graphstorm-processing'.
-v, --version   Docker version tag, default is the library's current version (`poetry version --short`)
-b, --build     Docker build directory, default is '/tmp/`
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
  TARGET='test'

  while :; do
    case "${1-}" in
    -h | --help) usage ;;
    -x | --verbose) set -x ;;
    --no-color) NO_COLOR=1 ;;
    -t | --target)
      TARGET="${2-}"
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
    -?*) die "Unknown option: $1" ;;
    *) break ;;
    esac
    shift
  done

  args=("$@")

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

if [[ ${TARGET} == "prod" || ${TARGET} == "test" ]]; then
    :  # Do nothing
else
    die "target parameter needs to be one of 'prod' or 'test', got ${TARGET}"
fi

# script logic here
msg "Execution parameters:"
msg "- TARGET: ${TARGET}"
msg "- GSP_HOME: ${GSP_HOME}"
msg "- IMAGE_NAME: ${IMAGE_NAME}"
msg "- VERSION: ${VERSION}"

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
    rsync -r ${GSP_HOME} "${BUILD_DIR}/docker/code/graphstorm-processing/" --exclude .venv --exclude dist
    cp ${GSP_HOME}/../graphstorm_job.sh "${BUILD_DIR}/docker/code/"
fi

# Copy Docker entry point to build folder
cp ${GSP_HOME}/docker-entry.sh "${BUILD_DIR}/docker/code/"

DOCKER_FULLNAME="${IMAGE_NAME}:${VERSION}"

# Login to ECR to be able to pull source SageMaker image
aws ecr get-login-password --region us-west-2 \
    | docker login --username AWS --password-stdin 153931337802.dkr.ecr.us-west-2.amazonaws.com

echo "Build a Docker image ${DOCKER_FULLNAME}"
DOCKER_BUILDKIT=1 docker build -f "${GSP_HOME}/docker/${VERSION}/Dockerfile.cpu" \
    "${BUILD_DIR}/docker/" -t $DOCKER_FULLNAME --target ${TARGET}
