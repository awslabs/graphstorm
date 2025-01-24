#!/bin/bash

date
echo "Args: $@"
env
echo "jobId: $AWS_BATCH_JOB_ID"
echo "jobQueue: $AWS_BATCH_JQ_NAME"
echo "computeEnvironment: $AWS_BATCH_CE_NAME"

COMMAND=$1
SOURCE_REF=$2
REMOTE=$3
# WORK_DIR=$2
# VAR1="chmod +x ./pytest_check.sh && ./pytest_check.sh"
# VAR2="chmod +x ./lint_check.sh && ./lint_check.sh"

git clone https://github.com/awslabs/graphstorm.git

cd graphstorm

echo "REMOTE: $REMOTE"
echo "SOURCE-REF: $SOURCE_REF"

if [ ! -z $REMOTE ]; then
    git remote set-url origin $REMOTE
fi

git fetch origin $SOURCE_REF:working
git checkout working

# cd $WORK_DIR
cd .github/workflow_scripts

#Carriage removal
COMMAND=`sed -e 's/^"//' -e 's/"$//' <<<"$COMMAND"`

/bin/bash -o pipefail -c "$COMMAND"
COMMAND_EXIT_CODE=$?

echo "Exit code: $COMMAND_EXIT_CODE"
exit $COMMAND_EXIT_CODE
