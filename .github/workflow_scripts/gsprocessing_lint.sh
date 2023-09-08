# Move to parent directory
cd ../../

set -ex

echo "hello"
python3 -m pip install --upgrade prospector pip
FORCE_CUDA=1 python3 -m pip install -e '.[test]'  --no-build-isolation

pylint --rcfile=./tests/lint/pylintrc ./graphstorm-processing/graphstorm_processing/
