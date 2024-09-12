#!/usr/bin/env bash
set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

GS_HOME="/graphstorm"
cd /graphstorm/docker
echo $(pwd)

error_and_exit () {
	# check exec status of launch.py
	status=$1
	echo $status

	if test $status -ne 0
	then
		exit -1
	fi
}

df /dev/shm -h

echo "**************build docker image for GSF local image, platform: gpu"
bash build_docker_oss4local.sh GS_HOME gpu local
error_and_exit $?

echo "**************build docker image for GSF local image, platform: cpu"
bash build_docker_oss4local.sh GS_HOME cpu local
error_and_exit $?

echo "**************build docker image for GSF SageMaker, platform: gpu"
bash build_docker_sagemaker.sh GS_HOME gpu
error_and_exit $?

echo "**************build docker image for GSF SageMaker, platform: cpu"
bash build_docker_sagemaker.sh GS_HOME gpu
error_and_exit $?

echo "**************build docker image for GSF local parmetis, platform: cpu"
bash build_docker_parmetis.sh GS_HOME
error_and_exit $?
