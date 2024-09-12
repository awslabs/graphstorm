#!/usr/bin/env bash
set -Eeux

# Move to parent directory, where the repository was cloned
cd ../../

GS_HOME=$(pwd)
# Install graphstorm from checked out code
pip3 install "$GS_HOME" --upgrade

bash ./tests/end2end-tests/docker_build/docker_build.sh