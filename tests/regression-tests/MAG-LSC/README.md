## Graph partitioning
We perform graph partitioning on r6.32xlarge inside a dcoker container.

Run the docker container:

```
docker run -v $DATA_FOLDER/:/data -v $GS_HOME/:/graph-storm -v /dev/shm:/dev/shm --network=host -it 911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm_ci_nom5:v1.3 /bin/bash
```

Run the graph partitioning command.

```
python3 /graph-storm/tools/partition_graph_lp.py --dataset mag-lsc --num_parts 4 --balance_train --balance_edges --output /data/mag-lsc-4p --edge_pct 0.8 --filepath /data
```

## Distributed training
We perform distributed training on a cluster of g5.48xlarge instances with the following command.

The folder with regression tests provides a script to run docker containers on all machines in a cluster.
Before running the command below, please collect all IP addresses in the cluster and keep them in `ip_list.txt`, in which each row is an IP address.
Note: please change the data folder and the GraphStorm folder in `rerun_gsf_docker.sh` script.

```
bash $GS_HOME/tests/regression-tests/dist_run.sh ip_list.txt $GS_HOME/tests/regression-tests/rerun_gsf_docker.sh
```

After running the docker containers, attach to one of the docker container.

```
docker container exec -it regression_test /bin/bash
```

### Link prediction
Run the distributed link prediction on the MAG-LSC dataset inside the cluster of docker container.
Please be sure that the partitioned MAG-LSC dataset is under `/data` inside the docker containers.
```
bash /graph-storm/tests/regression-tests/MAG-LSC/run_mag_lsc_lp.sh
```

### Node classification
Run the distributed link prediction on the MAG-LSC dataset inside the cluster of docker container.
```
bash /graph-storm/tests/regression-tests/MAG-LSC/run_mag_lsc_nc.sh
```

Validation accuracy: 0.6656
