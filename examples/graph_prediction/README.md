# Example Code for Graph Prediction using GraphStorm
Graph prediction, such as classification and regression, is a common task in Graph Machine Learning (GML) across various domains, including life sciences and chemistry. In graph prediction, the entire graph data is typically organized in a batch of subgraphs format, where each subgraph's nodes have edges only within the subgraph and no edges connecting to nodes in other subgraphs. GML labels are linked to these subgraphs. And then GML model makes prediction on new subgraphs as well.

A typical operation used in graph prediction is called `Read-out`, e.g., `sum`, `mean`, or `max/min`, which aggregates the representations of nodes in a subgraph to form one representation for the subgraph. Then outputs of the `Read-out` will be used to perform downstream prediction computation.

The current version of GraphStorm can not directly perform graph prediction. But as GraphStorm supports node-level prediction, we can use a method called `super-node` to fulfill the graph prediction requirement. Instead of using the `Read-out` operation, we can add a new node, called **super node**, to each subgraph, and link all original nodes of the subgraph to it, but not adding reversed edges. With these inbound edges, representations of all original nodes in a subgraph could be easily aggregated to the **super node**. And then we can use the **super node** as the repsentation of this subgraph to perform graph level prediciton tasks. The `super-node` method help us to turn a graph prediction task into a node prediction task.

In this example, we demonstrate how to process the common graph property prediction dataset (the [OGBG datasets](https://ogb.stanford.edu/docs/graphprop/)) into the `super-node` format graph data. And then we can leverage GraphStorm's graph construciton, and model trainig CLIs to perform `super node` prediction for graph predicition tasks.

## `Super-node` Graph Data Processing

**Step 1**: Generate super-node format OGBG graph data.
``` bash
python gen_ogbg_supernode.py --ogbg-data-name molhiv \
                             --output-path ./
```

**Step 2**: Run GraphStorm graph construction CLI.
``` bash
python -m graphstorm.gconstruct.construct_graph \
        --conf-file ./config.json \
        --output-dir ./supernode_gs_1p/ \
        --num-parts 1 \
        --graph-name supernode_molhiv
```

## Traing GraphStorm GNN Models for `Super-node` Graphs

Using the `super-node` method, we turn the graph prediction task into a node prediction task. Then we can leverage GraphStorm's node prediction CLIs to perform the graph prediction on **super nodes**. Below is the CLI for training an RGCN model on the OGBG data.

``` bash
python -m graphstorm.run.gs_node_classification \
          --num-trainers 1 \
          --part-config ./supernode_gs_1p/supernode_molhiv.json \
          --cf supernode_gc.yaml \
          --save-model-path ./supernode_gc_model/
```

Users can find the training configuration file, `supernode_gc.yaml`, located at this folder.

We can also try out other GraphStorm GNN models with CLIs like the followings.

``` bash
# Use RGAT as the GNN model
python -m graphstorm.run.gs_node_classification \
          --num-trainers 1 \
          --part-config ./supernode_gs_1p/supernode_molhiv.json \
          --cf supernode_gc.yaml \
          --save-model-path ./supernode_gc_model/ \
          --model-encoder-type rgat \
          --num-heads 4
```

``` bash
# Use HGT as the GNN model
python -m graphstorm.run.gs_node_classification \
          --num-trainers 1 \
          --part-config ./supernode_gs_1p/supernode_molhiv.json \
          --cf supernode_gc.yaml \
          --save-model-path ./supernode_gc_model/ \
          --model-encoder-type hgt \
          --num-heads 4
```

> [!TIP]
> The built-in GNN models do not perform the same computation as the `Read-out` operation, which normally just aggregates the representations of nodes in their last GNN layer. To mimic the `Read-out` operation, we can create customize GraphStorm GNN encoders. To find more implementation details, users can refer to the [GraphStorm API programming examples](https://graphstorm.readthedocs.io/en/latest/api/notebooks/index.html).


> [!TIP]
> To help users to dive deep the super-node format graph structure, we also provide a dummy super-node graph generation script, i.e., the `dummy_supernode_data.py`. You can run the following commands to build a dummy super-node format graph dataset. This dummy data can also be used for debugging the super-node customized GraphStorm models.

``` bash
python dummy_supernode_data.py --num-subgraphs 200 \
                                --save-path ./dummy_raw/

python -m graphstorm.gconstruct.construct_graph \
        --conf-file ./dummy_raw/config.json \
        --output-dir ./dummy_gs_1p/ \
        --num-parts 1 \
        --graph-name dummy_supernode
```
