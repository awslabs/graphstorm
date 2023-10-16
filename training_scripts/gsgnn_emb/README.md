# Tutorial: Use GraphStorm for Generating Embeddings on Node

## 0. Pipeline Overview
GraphStorm supports to generate node embeddings only. It aims to develop a user-friendly command line interface, enabling users to efficiently generate Graph Neural Network (GNN) embeddings for all nodes in a graph, enhancing accessibility and usability for tasks involving large-scale relational data analysis.

To generate node embeddings via the command line interface, users must prepare a Graphstorm model and a yaml configuration identical to the one used in training. Users can choose between mini-batch inference or full-graph inference, with the resulting command providing node embeddings either for a specified target node or for the entire graph if no specific node is designated.

## 1. Required Input
Typically, the GraphStorm command line for generating node embeddings will appear as follows:
```bash        
python3 -m graphstorm.run.gs_gen_node_embedding \
        --workspace /tmp/ogbn-arxiv-nc \
        --num-trainers 4 \
        --part-config /tmp/ogbn_arxiv_nc_train_val_1p_4t/ogbn-arxiv.json \
        --ip-config /tmp/ogbn-arxiv-nc/ip_list.txt \
        --ssh-port 2222 \
        --save-embed-path /tmp/saved_embed \
        --restore-model-path /tmp/epoch-0 \
        --cf /graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml \
        --use-mini-batch-infer true \
        --logging-file /tmp/log.txt
```

In addition to the required input on training, there are some other input necessary when generating node embeddings:

### 1.1. Save Embed Path
Users need to specify the save-embed-path. Here is an example in yaml file:
```
---
version: 1.0
gsf:
  ...
  output:
    save_embed_path: <user_specified_saved_embed_path>
```

### 1.2 Restore Model Path
Users need to specify the restore model path. Here is an example in yaml file:
```
---
version: 1.0
gsf:
  ...
  output:
    restore-model-path: <user_specified_restore_model_path>
```

## 2. Expected Output
The output of gs_gen_node_embedding should have same result as the one used in the training command, here is an example:
```
emb_info.json           
node_emb.part00000.bin 
node_emb.part00001.bin
...

```
## 3. Note about Generating Embedding:
gs_gen_node_embedding will only generate meaningful embeddings for target nodes that are well trained during training:

* Node class/regression task: The model only save node embeddings of nodes with node types from [target_ntype](https://github.com/awslabs/graphstorm/blob/8163a084d84db2ca95796273f52ecf1b0e478010/docs/source/configuration/configuration-run.rst#node-classificationregression-specific). If there is no definition of target_ntype, gs_gen_node_embedding will generate node embedding on full graph.
* Edge class/regression task: The model only save node embeddings of nodes with node types from [target_etype](https://github.com/awslabs/graphstorm/blob/8163a084d84db2ca95796273f52ecf1b0e478010/docs/source/configuration/configuration-run.rst#node-classificationregression-specific). For target_etype, gs_gen_node_embedding will generate node embedding on node type defined in the target_etype. If there is no definition of target_etype, gs_gen_node_embedding will generate node embedding on full graph.
* link prediction task: The model will save node embeddings on full graph. It is different from how we save embeddings during training, to avoid the leakage of link information, we conduct training only on the training edges during training. However, things are different for node embedding generation, it is more intuitive to use all edges.



