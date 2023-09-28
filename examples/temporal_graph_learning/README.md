# Customized TGAT model for snapshot-based temporal graph learning

This folder contains Python codes that demonstrate how to leverage the GraphStorm Framework (GSF) for production-level snapshot-based temporal graph data. This example utilizes the TGAT as a GNN model and the MAG graph data.

## How to run this example

The following commands run within the GraphStorm docker environment, and assume the user is in the directory `graphstorm/examples/temporal_graph_learning/`.

**Step 1: Prepare the MAG dataset for using the GraphStorm.**

Download dataset
```
mkdir ./DATA
wget -P ./DATA/MAG https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/node_features.pt
wget -P ./DATA/MAG https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/labels.csv
wget -P ./DATA/MAG https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/edges.csv
```

**Step 2: Pre-processing the raw data into GraphStorm/DGL's data structure**

Run the command to convert the data into the required raw format by splitting the edges by time:
```
python3 gen_graph.py
```

Once the command is successfully executed, it will generate a set of folders and files within the `./DATA` folder.
These files contain pre-processed graph data that can be recognized and utilized by GraphStorm's graph construction tool. The generated files include:
- `edge-paper-cite_{time}-paper.parquet` files for edge type "paper-cite-paper" that contain interactions between nodes for each day
- `node-paper-{parquet_split}.parquet` files for node type "paper" and contain node features. These node features are divided into multiple parquet splits to comply with parquet's maximum row requirements.
- `author_{train/val/test}_idx.json` which specify the indices for the training, validation, and test sets.
- `partition_config.json` which store the file paths for the aforementioned files.


Afterward, you can use the graph construction tool to create the partitioned graph data with the command below:

```
python3 -m graphstorm.gconstruct.construct_graph \
           --conf-file ./DATA/partition_config.json \
           --output-dir ./DATA/MAG_Temporal \
           --num-parts 1 \
           --graph-name MAG_Temporal
```

Please note that we modified the downloaded data to undirected graph and removed all duplicated edges in `gen_graph.py`, which could lead to the best result on this dataset. Therefore, we do not have to explicitly call `--add-reverse-edges` in the second step.

**Step 3: Run the modified TGAT model**

Run the command below to train the modified TGAT model with GraphStorm for node classification task:

```
export WORKSPACE=/home/ubuntu/graphstorm/examples/temporal_graph_learning/

python3 -m graphstorm.run.launch \
    --workspace $WORKSPACE \
    --part-config $WORKSPACE/DATA/MAG_Temporal/MAG_Temporal.json \
    --ip-config ./ip_config.txt \
    --num-trainers 1 \
    --num-servers 1 \
    --num-omp-threads 1 \
    --num-samplers 4 \
    --ssh-port 2222 \
    main_nc.py \
    --cf ./graphstorm_train_script_nc_config.yaml \
    --save-model-path $WORKSPACE/model
```

The task is to predict the primary subject areas of the given arXiv papers, which is cast as an ordinary multi-class classification problem. The metric is the classification accuracy.

# Implementation

This example code provides time encoding and temporal aggregation based on GraphStorm's RGAT implementation.

- Time encoding is implemented as `field_embeds` in `NodeEncoderInputLayer`. `field_embeds` is a set of trainable embeddings for each timestamps. The corresponding embedding for each timestamp is selected from `field_embeds` and then added to the original node features to help the neural network in distinguishing between different timestamps.

- Temporal aggregation is implemented within `TemporalRelationalGraphEncoder`. This ensures that hidden embeddings with time $t$ are only computed by aggregating input embeddings with time $t^\prime \leq t$.

In this implementation, the node hidden embeddings at different timestamps are stored as the following format
```
embeds = {
    node_type_1: {
        'time_stamp_1': torch.Tensor,
        'time_stamp_2': torch.Tensor,
                    ...
        'time_stamp_t': torch.Tensor,
    }
}
```

# Performance

Validation accuracy: ~0.6451; Test accuracy: ~0.6486 on the customized MAG data