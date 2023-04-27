# Customized HGT model to use the GraphStorm Framework

This folder contains artifacts, Python codes and shell scripts, that demonstrate how to change a GNN model and its graph data to leverage the GraphStorm Framework (GSF) for production level graph data.

This example use the Heterogeneous Graph Transformer (HGT) as GNN model and the ACM graph data, which originally implemented by the DGL team in the [HGT DGL example](https://github.com/dmlc/dgl/tree/master/examples/pytorch/hgt). In this example, we made a few changes to handle more general cases, including:

- Change to use mini-batch train/inference mode. The original HGT DGL example uses a full graph train/inference mode. To scale it for extremely large graphs that the GSF is good at, we need to modify the model's forward() function to accept blocks (now called MFGs). Users can refer to the [DGL User Guide Chapter 6](https://docs.dgl.ai/en/1.0.x/guide/minibatch.html) to learn how to implement this change.
- Change the model to handle featureless nodes. For featureless node type, this example HGT use type-specific trainable embedding, i.e., all nodes in the same node type share the same embedding.

In order to plus users' own GNN models into the GraphStorm Framework, users need to perform two major modificationsas demonstrated in example. For detailed instruction of modifying your GNN models to fit into the GraphStorm Framework, please refer to [Customize GNN models for using GraphStorm Tutorial](https://w.amazon.com/bin/view/AWS/AmazonAI/AIRE/GSF/UseYourOwnGnnModels/).

1. Preprocess your graph data into DGL graphs and then use GSF's partition graph tools, e.g., [partition_graph.py](https://gitlab.aws.dev/agml/graph-storm/-/blob/opensource_gsf/tools/partition_graph.py) or [partition_graph_lp.py](https://gitlab.aws.dev/agml/graph-storm/-/blob/opensource_gsf/tools/partition_graph_lp.py), to convert the graph data into a distributed format that the GSF uses.

2. Convert the HGT model to use the GraphStorm customer GNN model APIs, including but not limited to:
    - Heritate your model from GraphStorm's base model, such as the [GSgnnNodeModelBase](https://gitlab.aws.dev/agml/graph-storm/-/blob/opensource_gsf/python/graphstorm/model/node_gnn.py#L56) or [GSgnnEdgeModelBase](https://gitlab.aws.dev/agml/graph-storm/-/blob/opensource_gsf/python/graphstorm/model/edge_gnn.py#L60), etc.
    - Implement the forward(), predict(), and create_optimizer() methods following the given API's requirements.
    - Define your own loss function, or use GraphStorm's built-in loss functions that can handel common classification, regression, and link predictioin tasks.
    - In case having unused weights problem, modify the loss computation to include a regulation computation of all parameters

3. Use the GraphStorm's dataset, e.g., [GSgnnNodeTrainData](https://gitlab.aws.dev/agml/graph-storm/-/blob/opensource_gsf/python/graphstorm/data/dataset.py#L8) and dataloader, e.g., [GSgnnNodeDataLoader](https://gitlab.aws.dev/agml/graph-storm/-/blob/opensource_gsf/python/graphstorm/dataloading/dataloading.py#L446) to construct distributed graph loading and mini-batch sampling.

4. Wrap your model in a GraphStorm trainer, e.g., [GSgnnNodePredictionTrainer](https://gitlab.aws.dev/agml/graph-storm/-/blob/opensource_gsf/python/graphstorm/trainer/np_trainer.py#L13), which will handle the training process with its fit() method.

## How to run this example
---------------------------
*Note:* The following commands run within the GraphStorm docker environment. And there should be a folder, "/data", in the docker environment.

**Step 1: Prepare the ACM dataset for using the GraphStorm**
```shell
python3 /graphstorm/examples/acm_data.py --output-path /data
```

**Step 2: Partition the ACM graph into distributed format**
```shell
python3 /graphstorm/tools/partition_graph.py \
    --dataset acm\
    --filepath /data \
    --num_parts 1 \
    --target_ntype paper \
    --nlabel_field paper:label \
    --output /data/acm_nc
```

**Step 3: Run the modified HGT model**
```shell
python3 ~/dgl/tools/launch.py \
    --workspace /graphstorm/examples/customized_models/HGT \
    --part_config /data/acm_nc/acm.json \
    --ip_config ip_list.txt \
    --num_trainers 2 \
    --num_servers 1 \
    --num_samplers 0 \
    --ssh_port 2222 \
    "python3 hgt_nc.py --yaml-config-file acm_nc.yaml \
                       --ip-config ip_list.txt \
                       --node-feat paper:feat-author:feat-subject:feat \
                       --num-heads 8"
```

## HGT+GSF Performance on the ACM data
-----------------------------------------
Validation accuracy: 0.4700; Test accuracy: 0.4664

DGL example Test accuracy: 0.465±0.007