# Move to parent directory
cd ../../

set -ex

sh ./tests/end2end-tests/setup.sh
sh ./tests/end2end-tests/create_data.sh
bash ./tests/end2end-tests/graphstorm-lp/mgpu_test.sh
bash ./tests/end2end-tests/graphstorm-nc/mgpu_test.sh
bash ./tests/end2end-tests/graphstorm-ec/mgpu_test.sh

