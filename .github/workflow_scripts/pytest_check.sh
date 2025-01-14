#!/bin/env bash
# Move to repository root
set -ex

cd ../../


GS_HOME=$(pwd)
# Add SageMaker launch scripts to make the scripts testable
export PYTHONPATH="${PYTHONPATH}:${GS_HOME}/sagemaker/launch/"

python3 -m pip install pytest
FORCE_CUDA=1 python3 -m pip install -e '.[test]'  --no-build-isolation

# Run SageMaker tests
python3 -m pytest -x ./tests/sagemaker-tests -s

# Run main library unit tests (Requires multi-gpu instance)
sh ./tests/unit-tests/prepare_test_data.sh
export NCCL_IB_DISABLE=1; export NCCL_SHM_DISABLE=1; NCCL_NET=Socket NCCL_DEBUG=INFO python3 -m pytest -x ./tests/unit-tests -s
