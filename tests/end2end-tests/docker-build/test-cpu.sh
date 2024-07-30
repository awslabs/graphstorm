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

echo "**************local docker image for cpu"
bash ./build_docker_oss4local.sh ~/graphstorm/ graphstorm latest cpu

error_and_exit $?

echo "**************local docker image for cpu, Parmetis container for EC2 Clusters"
bash ./build_docker_parmetis.sh ~/graphstorm/ graphstorm parmetis-cpu

error_and_exit $?

echo "**************local docker image for cpu, SageMaker"
bash ./build_docker_sagemaker.sh ~/graphstorm cpu

error_and_exit $?

echo 'Done'
