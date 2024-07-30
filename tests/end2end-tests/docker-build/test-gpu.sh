#!/bin/bash

service ssh restart

GS_HOME=$(pwd)
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/docker/

error_and_exit () {
	# check exec status of launch.py
	status=$1
	echo $status

	if test $status -ne 0
	then
		exit -1
	fi
}

echo "Test GraphStorm docker build"

date

pwd

echo "**************local docker image for gpu"
bash ./build_docker_oss4local.sh ~/graphstorm/ graphstorm latest gpu

error_and_exit $?

echo "**************local docker image for gpu, SageMaker"
bash ./build_docker_sagemaker.sh ~/graphstorm gpu

error_and_exit $?

echo 'Done'
