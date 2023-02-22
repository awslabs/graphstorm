#!/bin/bash

# This is the entrypoint script for AWS Batch
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


COMMAND=`sed -e 's/^"//' -e 's/"$//' <<<"$COMMAND"`

/bin/bash -o pipefail -c "$COMMAND"


echo "Test Complete"
exit



# -----------Test for carriage removal-------------
# pwd
# echo *
# echo "This is cmd with {} and $ outside"
# echo ${COMMAND}

# echo "This is cmd with $"
# echo $COMMAND

# echo "This is cmd with $ and quotes"
# echo "$COMMAND"

# echo "This is cmd without quotes and $"
# echo COMMAND

# echo "This is cmd with $ and without quotes"
# echo "$COMMAND"

# COMMAND=`sed -e 's/^"//' -e 's/"$//' <<<"$COMMAND"`
# echo "This is new cmd without quotes"
# echo $COMMAND

# echo "This is new cmd with quotes"
# echo "$COMMAND"

