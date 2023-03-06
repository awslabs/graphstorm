# Move to parent directory
cd ../../

python3 -m pip install --upgrade prospector pip
FORCE_CUDA=1 python3 -m pip install -e '.[test]'  --no-build-isolation
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/data/*.py
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/dataloading/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/config/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/eval/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/model/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/trainer/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/inference/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/tracker/
pylint --rcfile=./tests/lint/pylintrc ./python/graphstorm/utils.py
pylint --rcfile=./tests/lint/pylintrc ./sagemaker/launch_train.py
pylint --rcfile=./tests/lint/pylintrc ./sagemaker/scripts/*.py

