## GraphStorm
|[Document and Tutorial Site](https://github.com/awslabs/graphstorm/wiki) |

GraphStorm is a graph machine learning (GML) framework for enterprise use cases.
It simplifies the development, training and deployment of GML models for industry-scale graphs
by providing scalable training and inference pipelines of Graph Machine Learning (GML) models
for extremely large graphs (measured in billons of nodes and edges).
GraphStorm provides a collection of built-in GML models and users can train a GML model
with a single command without writing any code. To help develop SOTA models,
GraphStorm provides a large collection of configurations for customizing model implementations
and training pipelines to improve model performance. GraphStorm also provides a programming
interface to train any custom GML model in a distributed manner. Users
provide their own model implementations and use GraphStorm training pipeline to scale.

![GraphStorm architecture](https://github.com/awslabs/graphstorm/blob/main/tutorial/graphstorm_arch.jpg?raw=true)

## Get Started
### Installation
GraphStorm is compatible to Python 3.7+. It requires PyTorch 1.13+, DGL 1.0 and transformers 4.3.0+.

To install GraphStorm in your environment, you can clone the repository and run `python3 setup.py install` to install it. However, running GraphStorm in a distributed environment is non-trivial. Users need to install dependencies and configure distributed Pytorch running environments. For this reason, we highly recommend users to using [Docker](https://docs.docker.com/get-started/overview/) container to run GraphStorm. A guideline to build GraphStorm docker image and run it on Amazon EC2 can be found at [here](https://github.com/awslabs/graphstorm/tree/main/docker).

### Run GraphStorm with OGB datasets

**Note**: we assume users have setup their Docker container following the [Build a Docker image from source](https://github.com/awslabs/graphstorm/tree/main/docker#build-a-docker-image-from-source) instructions. All following commands run within a Docker container.

**Start the GraphStorm docker container**
First, start your docker container by running the following command:

```nvidia-docker run --network=host -v /dev/shm:/dev/shm/ -d --name test <graphstomr-image-name>```

After running the container as a daemon, you need to connect to your container:

```docker container exec -it test /bin/bash```

**Node classification on OGB arxiv graph**
First, use the below command to download the [OGB arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) data and process it into a DGL graph for the node classification task.

```python3 /graphstorm/tools/gen_ogb_dataset.py --savepath /tmp/ogbn-arxiv-nc/ --retain-original-features true```

Second, use the below command to partition this arxiv graph into a distributed graph that GraphStorm can use as its input.

```
python3 /graphstorm/tools/partition_graph.py --dataset ogbn-arxiv \
                                             --filepath /tmp/ogbn-arxiv-nc/ \
                                             --num-parts 1 \
                                             --num-trainers-per-machine 4 \
                                             --output /tmp/ogbn_arxiv_nc_train_val_1p_4t
```

GraphStorm distributed training relies on ssh to launch training jobs. These containers run ssh services in port 2222. Users need to collect the IP addresses of all machines and put all IP addresses in an ip_list.txt file, in which every row is an IP address. We suggest users to provide the ip_list.txt file’s absolute path in the launch script. If run GraphStorm training in a single machine, the ip_list.txt only contains one row as below.

```127.0.0.1```

NOTE: please do *NOT* leave blank lines in the ip_list.txt.

Third, run the below command to train an RGCN model to perform node classification on the partitioned arxiv graph.

```
python3 -m graphstorm.run.gs_node_classification \
        --workspace /tmp/ogbn-arxiv-nc \
        --num-trainers 1 \
        --num-servers 1 \
        --num-samplers 0 \
        --part-config /tmp/ogbn_arxiv_nc_train_val_1p_4t/ogbn-arxiv.json \
        --ip-config  /tmp/ogbn-arxiv-nc/ip_list.txt \
        --ssh-port 2222 \
        --cf /graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml \
        --save-perf-results-path /tmp/ogbn-arxiv-nc/models
```

**Link Prediction on OGB MAG graph**
First, use the below command to download the [OGB MAG](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) data and process it into a DGL graph for the link prediction task. The edge type for prediction is “*author,writes,paper*”. The command also set 80% of the edges of this type for training and validation (default 10%), and the rest 20% for testing.

```
python3 /graphstorm/tools/gen_mag_dataset.py --savepath /tmp/ogbn-mag-lp/ --edge-pct 0.8
```

Second, use the following command to partition the MAG graph into a distributed format.

```
python3 /graphstorm/tools/partition_graph_lp.py --dataset ogbn-mag \
                                                --filepath /tmp/ogbn-mag-lp/ \
                                                --num-parts 1 \
                                                --num-trainers-per-machine 4 \
                                                --target-etypes author,writes,paper \
                                                --output /tmp/ogbn_mag_lp_train_val_1p_4t
```

Third, run the below command to train an RGCN model to perform link prediction on the partitioned MAG graph.

```
python3 -m graphstorm.run.gs_link_prediction \
        --workspace /tmp/ogbn-mag-lp/ \
        --num-trainers 1 \
        --num-servers 1 \
        --num-samplers 0 \
        --part-config /tmp/ogbn_mag_lp_train_val_1p_4t/ogbn-mag.json \
        --ip-config /tmp/ogbn-mag-lp/ip_list.txt \
        --ssh-port 2222 \
        --cf /graphstorm/training_scripts/gsgnn_lp/mag_lp.yaml \
        --node-feat-name paper:feat \
        --save-model-path /tmp/ogbn-mag/models \
        --save-perf-results-path /tmp/ogbn-mag/models
```

## Limitation
GraphStorm framework now supports using CPU or NVidia GPU for model training and inference. But it only works with PyTorch-gloo backend. It was only tested on AWS CPU instances or AWS GPU instances equipped with NVidia GPUs including P4, V100, A10 and A100.

Multiple samplers are not supported for PyTorch versions greater than 1.12. Please use `--num-samplers 0` when your PyTorch version is above 1.12. You can find more details [here](https://github.com/awslabs/graphstorm/issues/199).

## License
This project is licensed under the Apache-2.0 License.


