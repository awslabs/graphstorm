# Move to parent directory
cd ../../

set -ex

python3 -m pip install autogluon.tabular[mitra]==1.4.0 einops==0.8.1

sh ./tests/end2end-tests/setup.sh
sh ./tests/end2end-tests/create_data.sh
sh ./tests/end2end-tests/custom-gnn/run_test.sh
bash ./tests/end2end-tests/graphstorm-nc/test-cpu.sh
bash ./tests/end2end-tests/graphstorm-lp/test-cpu.sh
bash ./tests/end2end-tests/graphstorm-ec/test-cpu.sh

kill $(jobs -p) 2>/dev/null || true; exit 0
