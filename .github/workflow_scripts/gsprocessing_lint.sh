# Move to parent directory
cd ../../

set -ex

pip install pylint==2.17.5

pylint --rcfile=./tests/lint/pylintrc ./graphstorm-processing/graphstorm_processing/
black --check ./graphstorm-processing/graphstorm_processing/
