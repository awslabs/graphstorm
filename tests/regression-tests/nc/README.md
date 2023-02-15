# Regression Tests for Node Classification

Need to make sure the performance of GraphStorm is the similar on the various node classification tasks after each major development. 

*note*: the regression test is designed to run on GraphStorm docker environment. To learn how to configure your Linux environment to run the GraphStorm in docker, please refer to the [GraphStorm Onboarding Tutorial](https://w.amazon.com/bin/view/AWS/AmazonAI/AIRE/GSF/OnboardTutorial). All below commands run within the GraphStorm docker container.

Prerequist:
-----------
1. The new rgnn_base.py codes need nightly build of DGL. So need to uninstall the v0.9.0 DGL first and reinstall the nightly build.
```shell
pip uninstall dgl-cu113
pip install --pre dgl-cu113 -f https://data.dgl.ai/wheels-test/repo.html
```

2. Make sure that the graph-storm codes are located in the /graph-storm folder within the docker environment.

3. set the GraphStorm Python path
```shell
export PYTHONPATH=/graph-storm/python
```

4. Configure the docker environment to allow it to access the target S3 bucket so that regression results could be saved to S3. For example, you can copy and paste the AK/SK bash command and run them in the container command line.

## OGB Arxiv Node Classification

How to run:
-----------
Step 1: cd to the graph-storm folder

Step 2: create test data
```shell
bash tests/regression-tests/nc/prepare_ogbn_arxiv_nc_regression_test_data.sh
```
Step 3: run the test
```shell
bash tests/regression-tests/nc/ogbn_arxiv_nc_regression_test.sh
```

Regression performance results:
-------------------------------
With two-RGCN layers, the best performance is 
```python
best_test_score: {'accuracy': 0.6305372096372652}
best_val_score: {'accuracy': 0.6522366522366523}
peak_mem_alloc_MB: 57.6025390625
```

## OGB-MAG Node Classification

Prepare test data:
------------------

Step 1: cd to the graph-storm folder
```shell
cd /path-to/graph-storm/
```

Step 2: run prepare shell script
```shell
sh -u tests/regression-tests/nc/prepare_ogbn_mag_nc_regression_test_data.sh
```

Step 3: run the regression test shell script
```shell
sh -u tests/regression-tests/nc/ogbn_mag_nc_regression_test.sh
```

Best performance:
-----------------
*note*: we run the regression tests on an g4dn.12xlarge EC2 instance with 4 GPUs.

### 4 GPUs
Epoch 00004, Train accuracy: unknown | Val accuracy: 0.4502 | Test accuracy: 0.4145, Eval time: 45.3539s

### 1 GPU
Epoch 00003, Train accuracy: unknown | Val accuracy: 0.4717 | Test accuracy: 0.4337, Eval time: 154.6369