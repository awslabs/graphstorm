## Graph partitioning
We perform graph partitioning on r6.32xlarge inside a dcoker container.

Run the docker container:

```
docker run -v $DATA_FOLDER/:/data -v $GS_HOME/:/graph-storm -v /dev/shm:/dev/shm --network=host -it 911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm_alpha:v3 /bin/bash
```

Run the graph partitioning command.

```
python3 /graph-storm/tools/partition_graph_lp.py --dataset ogbn-papers100M --num_parts 4 --balance_train --balance_edges --output /data/ogbn-papers100M-4p --edge_pct 0.8 --filepath /data
```

## Distributed training
We perform distributed training on a cluster of g5.48xlarge instances with the following command.

The previous step partitions a graph. The partitioned graph is uploaded to `s3://gsf-regression/ogbn-papers100M-4p`.
We can download the partitioned graph from the S3 bucket directly, instead of partitioning it every time.
The graph partition takes a long time. To download the partitioned graph from the S3 bucket, please set up
the credential of the dgl-pa account.

```
aws s3 cp --recursive s3://gsf-regression/ogbn-papers100M-4p $DATA_FOLDER/ogbn-papers100M-4p
```

Set up the cluster as described in `tests/regression-tests/README.md`.
After running the docker containers, attach to one of the docker container on one of the machines.

```
docker container exec -it regression_test /bin/bash
```

### Link prediction
Run the distributed link prediction on the OGBN-papers100M dataset inside the cluster of docker container.
Please be sure that the partitioned OGBN-papers100M dataset is under `/data` inside the docker containers.
```
bash /graph-storm/tests/regression-tests/OGBN-papers100M/run_papers100M_lp.sh
```

### Node classification
Run the distributed link prediction on the OGBN-papers100M dataset inside the cluster of docker container.
```
bash /graph-storm/tests/regression-tests/OGBN-papers100M/run_papers100M_nc.sh
```
