## GraphStorm
| [Document and Tutorial Site](https://graphstorm.readthedocs.io/en/latest/) | [GraphStorm Paper](https://arxiv.org/abs/2406.06022) |

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

GraphStorm can be installed with pip and it can be used to train GNN models in a standalone mode. To run GraphStorm in a distributed environment, we recommend users to using [Docker](https://docs.docker.com/get-started/overview/) container to reduce envrionment setup efforts. A guideline to setup GraphStorm running environment can be found at [here](https://graphstorm.readthedocs.io/en/latest/install/env-setup.html#setup-graphstorm-docker-environment) and a full instruction on how to setup distributed training can be found [here](https://graphstorm.readthedocs.io/en/latest/scale/distributed.html).

### Run GraphStorm with OGB datasets

**Note**: we assume users have setup a GraphStorm standalone environment following the [Setup GraphStorm with pip Packages](https://graphstorm.readthedocs.io/en/latest/install/env-setup.html#setup-graphstorm-with-pip-packages) instructions. And users have git cloned the GraphStorm source code into the `/graphstorm/` folder to use some complimentatry tools.

**Node classification on OGB arxiv graph**
First, use the below command to download the [OGB arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) data and process it into a DGL graph for the node classification task.

```
python /graphstorm/tools/gen_ogb_dataset.py --savepath /tmp/ogbn-arxiv-nc/ --retain-original-features true
```

Second, use the below command to partition this arxiv graph into a distributed graph that GraphStorm can use as its input.

```
python /graphstorm/tools/partition_graph.py --dataset ogbn-arxiv \
                                            --filepath /tmp/ogbn-arxiv-nc/ \
                                            --num-parts 1 \
                                            --num-trainers-per-machine 4 \
                                            --output /tmp/ogbn_arxiv_nc_train_val_1p_4t
```

GraphStorm training relies on ssh to launch training jobs. The GraphStorm standalone mode uses ssh services in port 22.

Third, run the below command to train an RGCN model to perform node classification on the partitioned arxiv graph.

```
python -m graphstorm.run.gs_node_classification \
       --workspace /tmp/ogbn-arxiv-nc \
       --num-trainers 1 \
       --part-config /tmp/ogbn_arxiv_nc_train_val_1p_4t/ogbn-arxiv.json \
       --ssh-port 22 \
       --cf /graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml \
       --save-perf-results-path /tmp/ogbn-arxiv-nc/models
```

**Link Prediction on OGB MAG graph**
First, use the below command to download the [OGB MAG](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) data and process it into a DGL graph for the link prediction task. The edge type for prediction is “*author,writes,paper*”. The command also set 80% of the edges of this type for training and validation (default 10%), and the rest 20% for testing.

```
python /graphstorm/tools/gen_mag_dataset.py --savepath /tmp/ogbn-mag-lp/ --edge-pct 0.8
```

Second, use the following command to partition the MAG graph into a distributed format.

```
python /graphstorm/tools/partition_graph_lp.py --dataset ogbn-mag \
                                               --filepath /tmp/ogbn-mag-lp/ \
                                               --num-parts 1 \
                                               --num-trainers-per-machine 4 \
                                               --target-etypes author,writes,paper \
                                               --output /tmp/ogbn_mag_lp_train_val_1p_4t
```

Third, run the below command to train an RGCN model to perform link prediction on the partitioned MAG graph.

```
python -m graphstorm.run.gs_link_prediction \
       --workspace /tmp/ogbn-mag-lp/ \
       --num-trainers 1 \
       --num-servers 1 \
       --num-samplers 0 \
       --part-config /tmp/ogbn_mag_lp_train_val_1p_4t/ogbn-mag.json \
       --ssh-port 22 \
       --cf /graphstorm/training_scripts/gsgnn_lp/mag_lp.yaml \
       --node-feat-name paper:feat \
       --save-model-path /tmp/ogbn-mag/models \
       --save-perf-results-path /tmp/ogbn-mag/models
```

To learn GraphStorm's full capabilities, please refer to our [Documentations and Tutorials](https://graphstorm.readthedocs.io/en/latest/).


## Cite

If you use GraphStorm in a scientific publication, we would appreciate citations to the following paper:
```
@article{zheng2024graphstorm,
  title={GraphStorm: all-in-one graph machine learning framework for industry applications},
  author={Zheng, Da and Song, Xiang and Zhu, Qi and Zhang, Jian and Vasiloudis, Theodore and Ma, Runjie and Zhang, Houyu and Wang, Zichen and Adeshina, Soji and Nisa, Israt and others},
  journal={arXiv preprint arXiv:2406.06022},
  year={2024}
}
```


## Limitation
GraphStorm framework now supports using CPU or NVidia GPU for model training and inference. But it only works with PyTorch-gloo backend. It was only tested on AWS CPU instances or AWS GPU instances equipped with NVidia GPUs including P4, V100, A10 and A100.

Multiple samplers are supported in PyTorch versions <= 1.12 and >= 2.1.0. Please use `--num-samplers 0` for other PyTorch versions. More details [here](https://github.com/awslabs/graphstorm/issues/199).

To use multiple samplers on sagemaker please use PyTorch versions <= 1.12.

## License
This project is licensed under the Apache-2.0 License.


