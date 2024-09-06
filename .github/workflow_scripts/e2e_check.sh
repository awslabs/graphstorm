#!/usr/bin/env bash.
# Move to parent directory

cd ../../

set -ex

bash ./tests/end2end-tests/setup.sh
bash ./tests/end2end-tests/create_data.sh
bash ./tests/end2end-tests/tools/test_mem_est.sh
bash ./tests/end2end-tests/data_process/test.sh
bash ./tests/end2end-tests/data_process/movielens_test.sh
bash ./tests/end2end-tests/data_process/homogeneous_test.sh
bash ./tests/end2end-tests/custom-gnn/run_test.sh
bash ./tests/end2end-tests/graphstorm-nc/test.sh
bash ./tests/end2end-tests/graphstorm-lp/test.sh
bash ./tests/end2end-tests/graphstorm-ec/test.sh
bash ./tests/end2end-tests/graphstorm-er/test.sh
