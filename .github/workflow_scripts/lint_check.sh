# Move to parent directory
cd ../../

set -ex

pip install pylint==2.17.5
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/data/*.py
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/distributed/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/dataloading/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/gconstruct/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/gpartition/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/config/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/eval/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/model/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/trainer/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/inference/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/tracker/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/run/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/utils.py
pylint --rcfile=./tests/lint/pylintrc ./tools/convert_feat_to_wholegraph.py

pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/sagemaker/
