# Move to parent directory
cd ../../

pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com pylibwholegraph-cu11==24.4.0

set -ex

sh ./tests/end2end-tests/setup.sh
sh ./tests/end2end-tests/create_data.sh
bash ./tests/end2end-tests/graphstorm-lp/mgpu_test.sh
bash ./tests/end2end-tests/graphstorm-mt/mgpu_test.sh
