# Link Prediction regression tests

## MAG Link Prediction Regression Test
Original MAG graph is designed for node classification. In order to do link prediction, we set the "author,writes,paper" edge as the
target edge. Also for link prediction, we split these edges according to a percentage (0.8) for training and validation (8:2).

*Note*: the regression test is designed to run on GraphStorm docker environment. To learn how to configure your Linux environment to run the GraphStorm in docker, please refer to the [GraphStorm Onboarding Tutorial](https://w.amazon.com/bin/view/AWS/AmazonAI/AIRE/GSF/OnboardTutorial). All below commands run within the GraphStorm docker container.

Prerequists
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
sh -u tests/regression-tests/lp/prepare_ogbn_mag_lp_regression_test_data.sh
```

step 3: run the test
```shell
sh -u tests/regression-tests/lp/ogbn_mag_lp_regression_test.sh
```

Regression performance:
-----------------------
In the opensource_gsf branch, the MAG link prediction performans much better than those in both M5GNN and the GSF main branch. The best test mrr culd reach to >0.90 while this value in the main branch is less than 0.4.

GSF Opensource results:
-----------------------
Train mrr: 0.9597, Val mrr: 0.9278, Test mrr: 0.8989, 
Best val mrr: 0.9278, Best test mrr: 0.8989, Best iter: 3130

Train mrr: 0.9697, Val mrr: 0.9443, Test mrr: 0.9182,
Best val mrr: 0.9443, Best test mrr: 0.9182, Best iter: 6260

GSF results:
------------
Train mrr: 0.9988, Val mrr: 0.3039, Test mrr: 0.3953, 
Best val mrr: 0.3039, Best test mrr: 0.3953, Best iter: 3130

M5GNN results:
--------------
Train mrr: 0.9980, Val mrr: 0.2165, Test mrr: 0.2641, 
Best val mrr: 0.2429, Best test mrr: 0.2641, Best iter: 2190
