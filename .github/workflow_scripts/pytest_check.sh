# Move to parent directory
cd ../../

set -ex

FORCE_CUDA=1 python3 -m pip install -e '.[test]'  --no-build-isolation
python3 -m pip install pytest
sh ./tests/unit-tests/prepare_test_data.sh
export NCCL_IB_DISABLE=1; export NCCL_SHM_DISABLE=1; NCCL_NET=Socket NCCL_DEBUG=INFO python3 -m pytest -x ./tests/unit-tests -s

