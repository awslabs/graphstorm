# Move to parent directory
cd ../../

set -ex

sh ./tests/end2end-tests/setup.sh
sh ./tests/end2end-tests/create_data.sh
sh ./tests/end2end-tests/custom-gnn/run_test.sh
bash ./tests/end2end-tests/graphstorm-nc/test-cpu.sh
bash ./tests/end2end-tests/graphstorm-lp/test-cpu.sh
bash ./tests/end2end-tests/graphstorm-ec/test-cpu.sh
