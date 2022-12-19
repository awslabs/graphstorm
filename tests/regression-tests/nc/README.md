## Regression Tests for Node Classification

Need to make sure the performance of GraphStorm on the various node classification tasks. 

### Node Classification on Arxiv graph
*note*: the regression test is designed to run on GraphStorm docker environment. To learn how to configure your Linux environment to run the GraphStorm in docker, please refer to the [GraphStorm Onboarding Tutorial](https://w.amazon.com/bin/view/AWS/AmazonAI/AIRE/GSF/OnboardTutorial). All below commands run within the GraphStorm docker container.

Prerequist:
-----------
- set the GraphStorm Python path
```shell
export PYTHONPATH=/graph-storm/python
```

How to run:
-----------
Step 1: cd to the graph-storm folder

Step 2: create test data
```shell
bash tests/regression-tests/create_test_data.sh
```
Step 3: run the test
```shell
bash tests/regression-tests/nc/ogbn_arxiv_nc_regression_test.sh
```
TODO: will complete this pipeline once the results parser is ready.

Regression performance results:
-------------------------------
With two-RGCN layers, the best performance is 
```python
best_test_score: {'accuracy': 0.6305372096372652}
best_val_score: {'accuracy': 0.6522366522366523}
peak_mem_alloc_MB: 57.6025390625
```