# Move to parent directory
cd ../../

set -ex

bash ./tests/end2end-tests/setup.sh
bash ./tests/end2end-tests/create_data.sh
bash ./tests/end2end-tests/custom-gnn/run_test.sh
bash ./tests/end2end-tests/graphstorm-nc/test-cpu.sh
bash ./tests/end2end-tests/graphstorm-lp/test-cpu.sh
bash ./tests/end2end-tests/graphstorm-ec/test-cpu.sh
