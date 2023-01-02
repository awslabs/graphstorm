## Set up a network filesystem

To set up distributed training, it is better to set up a network filesystem and store all graph data and code in the network filesystem.
To make it simple, we can use NFS or EFS to set up network filesystem.
[Here](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage/dist#step-0-setup-a-distributed-file-system)
is the instruction of setting up NFS for a cluster.

For an EC2 cluster, we can set up EFS. [Here](https://docs.aws.amazon.com/efs/latest/ug/gs-step-two-create-efs-resources.html)
is the instruction of creating EFS; [here](https://docs.aws.amazon.com/efs/latest/ug/installing-amazon-efs-utils.html)
is the instruction of installing an EFS client; [here](https://docs.aws.amazon.com/efs/latest/ug/efs-mount-helper.html)
provides the instructions of mounting the EFS filesystem.

After setting up EFS/NFS, we can keep all graph data and the code in the network filesystem.
Suppose EFS/NFS is mounted in `EFS_FOLDER` and we keep the graph data in the following folders.

```
GS_HOME=$EFS_FOLDER/graph-storm
DATA_FOLDER=$EFS_FOLDER/gsf-data
```

## Set up a docker cluster

We set up an EC2 cluster with docker containers for distributed training. First, we need to download the docker image on all machines in the cluster.

```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 911734752298.dkr.ecr.us-east-1.amazonaws.com
docker pull 911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm_alpha:v3
```

The folder with regression tests provides a script `dist_run.sh` to run docker containers on all machines in a cluster.
Before running the command below, please collect all private IP addresses in the EC2 cluster and keep them in `ip_list.txt`, in which each row is an IP address.
Note: please change `DATA_FOLDER` and `GS_HOME` in `rerun_gsf_docker.sh` script accordingly.

```
bash $GS_HOME/tests/regression-tests/dist_run.sh ip_list.txt $GS_HOME/tests/regression-tests/rerun_gsf_docker.sh
```
