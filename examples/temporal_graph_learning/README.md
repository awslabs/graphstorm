# Customized TGAT model for snapshot-based temporal graph learning

This folder contains python codes that demonstrate how to leverage the GraphStorm Framework (GSF) for production level snapshot-based temporal graph data.
This example uses the TGAT as GNN model and the MAG graph data.

## How to run this example

The following commands assume user in directory `graphstorm/examples/temporal_graph_learning/`.

**Step 1: Prepare the MAG dataset for using the GraphStorm.**

Download dataset
```
mkdir ./DATA
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/node_features.pt
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/labels.csv
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edges.csv
```

**Step 2: Pre-processing the raw data into GraphStorm/DGL's data structure**

Run the command to create the data with the required raw format.
```
python3 gen_graph.py
```

Once succeeded, the command will create a set of folders and files under the `./DATA` folder.
Then we can use the tool to create the partition graph data with the following command.

```
python3 -m graphstorm.gconstruct.construct_graph \
           --conf-file ./DATA/partition_config.json \
           --output-dir ./DATA/MAG_Temporal \
           --num-parts 1 \
```

**Step 3: Run the modified TGAT model**

Generate the GraphStorm launch script `run_script.sh`, ip configs `ip_config.txt`, and model configs `graphstorm_train_script_nc_config.yaml` using python script.
```
python3 generate_launch_script_nc.py
```

Then run the below command to train the modified TGAT model with GraphStorm.
```
bash run_script.sh
```

# TGAT+GSF Performance

Validation accuracy: ~0.6135; Test accuracy: ~0.6104 after 50 epcohs on the customized MAG data