# Customized TGAT model for snapshot-based temporal graph learning

This folder contains Python codes that demonstrate how to leverage the GraphStorm Framework (GSF) for production-level snapshot-based temporal graph data. This example utilizes the TGAT as a GNN model and the MAG graph data.

## How to run this example

The following commands assume the user is in the directory `graphstorm/examples/temporal_graph_learning/`.

**Step 1: Prepare the MAG dataset for using the GraphStorm.**

Download dataset
```
mkdir ./DATA
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/node_features.pt
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/labels.csv
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edges.csv
```

**Step 2: Pre-processing the raw data into GraphStorm/DGL's data structure**

Run the command to convert the data into the required raw format:
```
python3 gen_graph.py
```

Once successful, the command will generate a set of folders and files under the `./DATA` folder, which contains pre-processed graph data recognizable by the GraphStorm's graph construction tool.
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

Generate the GraphStorm launch script `run_script.sh`, IP configurations `ip_config.txt`, and model configurations `graphstorm_train_script_nc_config.yaml` using the Python script:
```
python3 generate_launch_script_nc.py
```

Then run the command below to train the modified TGAT model with GraphStorm:
```
bash run_script.sh
```

# Implementation

This example code provides time encoding and temporal aggregation based on GraphStorm's RGAT implementation.

- Time encoding is implemented as `field_embeds` in `NodeEncoderInputLayer`. `field_embeds` is a set of trainable embeddings for each timestamps. The corresponding embedding for each timestamp is selected from `field_embeds` and then added to the original node features to help the neural network in distinguishing between different timestamps.

- Temporal aggregation is implemented within `TemporalRelationalGraphConv`. This ensures that hidden embeddings with time $t$ are only computed by aggregating input embeddings with time $t^\prime \leq t$.

# Performance

Validation accuracy: ~0.6451; Test accuracy: ~0.6486 on the customized MAG data