# Move to parent directory
cd ../../

set -ex

sh ./tests/end2end-tests/setup.sh
sh ./tests/end2end-tests/create_data.sh
sh ./tests/end2end-tests/tools/test_mem_est.sh
sh ./tests/end2end-tests/data_process/test.sh
sh ./tests/end2end-tests/data_process/movielens_test.sh
sh ./tests/end2end-tests/data_process/homogeneous_test.sh
sh ./tests/end2end-tests/custom-gnn/run_test.sh
bash ./tests/end2end-tests/graphstorm-nc/test.sh
bash ./tests/end2end-tests/graphstorm-lp/test.sh
bash ./tests/end2end-tests/graphstorm-ec/test.sh
bash ./tests/end2end-tests/graphstorm-er/test.sh
