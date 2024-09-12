#!/usr/bin/env bash
set -Eeux

# Move to parent directory, where the repository was cloned
cd ../../

GS_HOME=$(pwd)
echo $GS_HOME

bash ./tests/end2end-tests/docker_build/docker_build.sh
