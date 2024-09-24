#!/usr/bin/env bash
set -Eeux

# Move to parent directory, where the repository was cloned
cd ../../

GS_HOME=$(pwd)
# Install graphstorm from checked out code
pip3 install "$GS_HOME" --upgrade

bash ./tests/end2end-tests/create_data.sh
bash ./tests/end2end-tests/graphbolt-gs-integration/graphbolt-graph-construction.sh
bash ./tests/end2end-tests/graphbolt-gs-integration/graphbolt-training-inference.sh
