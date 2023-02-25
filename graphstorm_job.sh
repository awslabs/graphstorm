#!/bin/bash

# Description: 
# This is the entry point script for AWS Batch
# Batch will execute this file and all the commands in this shell script
# These commands will be executed in a docker container running on EC2 instance and it's image is present in AWS ECR 
date
echo "Args: $@"
env
echo "jobId: $AWS_BATCH_JOB_ID"
echo "jobQueue: $AWS_BATCH_JQ_NAME"
echo "computeEnvironment: $AWS_BATCH_CE_NAME"

# WORK_DIR=$2
COMMAND=$1
# VAR1="chmod +x ./pytest_check.sh && ./pytest_check.sh"
# VAR2="chmod +x ./lint_check.sh && ./lint_check.sh"


# cd $WORK_DIR
cd graph-storm/.github/workflow_scripts

# Carriage removal
COMMAND=`sed -e 's/^"//' -e 's/"$//' <<<"$COMMAND"`

/bin/bash -o pipefail -c "$COMMAND"


echo "Test Complete"
exit
