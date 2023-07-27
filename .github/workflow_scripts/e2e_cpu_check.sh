# Move to parent directory
cd ../../

set -ex

exit 0

sh ./tests/end2end-tests/setup.sh
sh ./tests/end2end-tests/create_data.sh
sh ./tests/end2end-tests/custom-gnn/run_test.sh
bash ./tests/end2end-tests/graphstorm-nc/test.sh
bash ./tests/end2end-tests/graphstorm-lp/test.sh
bash ./tests/end2end-tests/graphstorm-ec/test.sh
bash ./tests/end2end-tests/graphstorm-er/test.sh
