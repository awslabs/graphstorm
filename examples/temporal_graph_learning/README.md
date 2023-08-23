# Customized TGAT model for snapshot-based temporal graph learning

This folder contains python codes that demonstrate how to leverage the GraphStorm Framework (GSF) for production level snapshot-based temporal graph data.

This example uses the TGAT as GNN model and the MAG graph data.

(Details on model architecture and data statistics will come soon)

## How to run this example

The following commands assume user in directory `graphstorm/examples/temporal_graph_learning/`.

**
Step 1: Prepare the MAG dataset for using the GraphStorm.
**

Download dataset
```
mkdir ./DATA
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/node_features.pt
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/labels.csv
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edges.csv
```

**
Step 2: Pre-processing the raw data into GraphStorm/DGL's data structure
**

```
python3 gen_graph.py
```

**
Step 3: Run the modified TGAT model
**

Generate the bash run script `run_script.sh`, ip configs `ip_config.txt`, and model configs `graphstorm_train_script_nc_config.yaml` using python script.
```
python3 generate_launch_script_nc.py
```

Then run the below command to train the modified TGAT model with GraphStorm.
```
bash run_script.sh
```
