## GraphStorm: Enterprise graph machine learning framework for billion-scale graphs

[![PyPI version](https://badge.fury.io/py/graphstorm.svg)](https://pypi.org/project/graphstorm/#history)
[![CI Status](https://github.com/awslabs/graphstorm/actions/workflows/continuous-integration.yml/badge.svg)](https://github.com/awslabs/graphstorm/actions/workflows/continuous-integration.yml)
[![Docs Status](https://app.readthedocs.org/projects/graphstorm/badge/?version=latest)](https://graphstorm.readthedocs.io/en/latest/)

| [Documentation and Tutorial Site](https://graphstorm.readthedocs.io/en/latest/) | [GraphStorm Paper](https://arxiv.org/abs/2406.06022) |

GraphStorm is an enterprise-grade graph machine learning (GML) framework designed for scalability and ease of use.
It simplifies the development and deployment of GML models on industry-scale graphs with billions of nodes and edges.

GraphStorm provides a collection of built-in GML models and users can train a GML model
with a single command without writing any code. To help develop SOTA models,
GraphStorm provides a large collection of configurations for customizing model implementations
and training pipelines to improve model performance. GraphStorm also provides a programming
interface to train any custom GML model in a distributed manner. Users
provide their own model implementations and use GraphStorm training pipeline to scale.

## Key Features
- Single-command GML model training and inference
- Distributed training/inference on industry-scale graphs (billions of nodes/edges)
- Built-in model collection
- AWS integration out-of-the-box


![GraphStorm architecture](https://github.com/awslabs/graphstorm/blob/main/tutorial/graphstorm_arch.jpg?raw=true)

## Get Started


### Installation

GraphStorm is compatible with Python 3.8+. It requires PyTorch 1.13+, DGL 1.0+ and transformers 4.3.0+.
For a full quick-start example see the [GraphStorm documentation](https://graphstorm.readthedocs.io/en/latest/#getting-started).

You can install and use GraphStorm locally using pip:

```bash
# If running on CPU use
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install dgl==2.3.0 -f https://data.dgl.ai/wheels/torch-2.3/repo.html

# Or, to run on GPU use
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl==2.3.0+cu121 -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

pip install graphstorm
```

### Distributed training

To run GraphStorm in a distributed environment, we recommend using [Amazon SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html) to avoid having to manage cluster infrastructure. See our
[SageMaker AI setup documentation](https://graphstorm.readthedocs.io/en/latest/cli/model-training-inference/distributed/sagemaker.html) to get started with distributed GNN training.

## Quick start


After installing GraphStorm and its requirements in your local environment as shown above, you can clone the GraphStorm repository to follow along the quick start examples:

```bash
git clone https://github.com/awslabs/graphstorm.git
# Switch to the graphstorm repository root
cd graphstorm
```

### Node Classification on OGB arxiv graph

This example demonstrates how to train a model to classify research papers in the OGB arxiv citation network. Each node represents a paper with a 128-dimensional feature vector, and the task is to predict the paper's subject area.

First, download the [OGB arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) data and process it into a DGL graph for the node classification task.

```bash
python tools/partition_graph.py \
    --dataset ogbn-arxiv \
    --filepath /tmp/ogbn-arxiv-nc/ \
    --num-parts 1 \
    --output /tmp/ogbn_arxiv_nc_1p

```

Second, train an RGCN model to perform node classification on the partitioned arxiv graph.

```bash
# create the workspace folder
mkdir /tmp/ogbn-arxiv-nc

python -m graphstorm.run.gs_node_classification \
    --workspace /tmp/ogbn-arxiv-nc \
    --num-trainers 1 \
    --num-servers 1 \
    --part-config /tmp/ogbn_arxiv_nc_1p/ogbn-arxiv.json \
    --cf "$(pwd)/training_scripts/gsgnn_np/arxiv_nc.yaml" \
    --save-model-path /tmp/ogbn-arxiv-nc/models
```

Third, run inference using the trained model

```bash
python -m graphstorm.run.gs_node_classification \
          --inference \
          --workspace /tmp/ogbn-arxiv-nc \
          --num-trainers 1 \
          --num-servers 1 \
          --part-config /tmp/ogbn_arxiv_nc_1p/ogbn-arxiv.json \
          --cf "$(pwd)/training_scripts/gsgnn_np/arxiv_nc.yaml" \
          --save-prediction-path /tmp/ogbn-arxiv-nc/predictions/ \
          --restore-model-path /tmp/ogbn-arxiv-nc/models/epoch-7/
```


### Link Prediction on OGB arxiv graph

First, download the [OGB arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) data and process it into a DGL graph for a link prediction task. The edge type we are trying to predict is `author,writes,paper`.

```bash
python ./tools/partition_graph_lp.py --dataset ogbn-arxiv \
                                   --filepath /tmp/ogbn-arxiv-lp/ \
                                   --num-parts 1 \
                                   --output /tmp/ogbn_arxiv_lp_1p/
```

Second, train an RGCN model to perform link prediction on the partitioned graph.

```bash
mkdir /tmp/ogbn-arxiv-lp
python -m graphstorm.run.gs_link_prediction \
          --workspace /tmp/ogbn-arxiv-lp \
          --num-trainers 1 \
          --num-servers 1 \
          --part-config /tmp/ogbn_arxiv_lp_1p/ogbn-arxiv.json \
          --cf "$(pwd)/training_scripts/gsgnn_lp/arxiv_lp.yaml" \
          --save-model-path /tmp/ogbn-arxiv-lp/models \
          --num-epochs 2
```

Third, run inference to generate node embeddings that you can use to run node similarity queries

```bash
python -m graphstorm.run.gs_gen_node_embedding \
           --workspace /tmp/ogbn-arxiv-lp \
           --num-trainers 1 \
           --num-servers 1 \
           --part-config /tmp/ogbn_arxiv_lp_1p/ogbn-arxiv.json \
           --cf "$(pwd)/training_scripts/gsgnn_lp/arxiv_lp.yaml" \
           --save-embed-path /tmp/ogbn-arxiv-lp/embeddings/ \
           --restore-model-path /tmp/ogbn-arxiv-lp/models/epoch-1/
```

For more detailed tutorials and documentation, visit our [Documentation site](https://graphstorm.readthedocs.io/en/latest/).


## Citation

If you use GraphStorm in a scientific publication, we would appreciate citations to the following paper:

```
@inproceedings{10.1145/3637528.3671603,
author = {Zheng, Da and Song, Xiang and Zhu, Qi and Zhang, Jian and Vasiloudis, Theodore and Ma, Runjie and Zhang, Houyu and Wang, Zichen and Adeshina, Soji and Nisa, Israt and Mottini, Alejandro and Cui, Qingjun and Rangwala, Huzefa and Zeng, Belinda and Faloutsos, Christos and Karypis, George},
title = {GraphStorm: All-in-one Graph Machine Learning Framework for Industry Applications},
year = {2024},
url = {https://doi.org/10.1145/3637528.3671603},
doi = {10.1145/3637528.3671603},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {6356–6367},
location = {Barcelona, Spain},
series = {KDD '24}
}
```

## Blog posts

The GraphStorm team has published multiple blog posts with use-case examples and highlighting new GraphStorm features.
These can help new users use GraphStorm in their production use-cases:

* [Fast-track graph ML with GraphStorm: A new way to solve problems on enterprise-scale graphs](https://aws.amazon.com/blogs/machine-learning/fast-track-graph-ml-with-graphstorm-a-new-way-to-solve-problems-on-enterprise-scale-graphs/)
* [GraphStorm 0.3: Scalable, multi-task learning on graphs with user-friendly APIs](https://aws.amazon.com/blogs/machine-learning/graphstorm-0-3-scalable-multi-task-learning-on-graphs-with-user-friendly-apis/)
* [Mitigating risk: AWS backbone network traffic prediction using GraphStorm](https://aws.amazon.com/blogs/machine-learning/mitigating-risk-aws-backbone-network-traffic-prediction-using-graphstorm/)
* [Faster distributed graph neural network training with GraphStorm v0.4](https://aws.amazon.com/blogs/machine-learning/faster-distributed-graph-neural-network-training-with-graphstorm-v0-4/)


## Limitations

- Supports CPU or NVIDIA GPUs for training and inference
- Multiple samplers only supported in PyTorch versions >= 2.1.0

## License
This project is licensed under the Apache-2.0 License.
