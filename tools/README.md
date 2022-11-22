# Partition a graph for a link prediction task

Partition a graph for a semantic matching task

filepath is the graph path
Specify num_trainers_per_machine larger than 1
num_parts is the number of machines number of partitions
output is the name of the partioned output folder
```
python3 partition_graph_lp.py --dataset query_asin_match --filepath qa_data_graph_v_1/ --num_parts 8 --num_trainers_per_machine 8 --output qa_train_v01_8p
```

# Partition paper100m
```
python3 /fsx-dev/xiangsx/home/workspace/graph-storm/tools/partition_graph.py --dataset ogbn-papers100m --filepath ./paper100m-processed-512/ --num_parts 8 --predict_ntypes "node" --balance_train --num_trainers_per_machine 8 --output /fsx-dev/xiangsx/home/workspace/graph-storm/training_scripts/gsgnn_nc/ogbn_papers100m_nc_8p_8t/
```

# Estimate the memory requirement of M5GNN training and inference.

`m5gnn_mem_est.py` is a script that can estimate the memory requirement to train M5GNN models and inference M5GNN models on a given graph data.

To estimate the memory requirement for training, a user needs to specify the supervision task ('node' and 'edge').
This estimation is only for training. If the training script turns on model evaluation during the training, the memory estimation here does not apply to that case.
An example is shown below:

```
export PYTHONPATH=/path/to/graph-storm/python/
python3 m5gnn_mem_est.py --root_path data_folder --supervise_task edge --is_train 1
```

To estimate the memory requirement for inference, a user needs to specify the number of hidden dimensions, the number of GNN layers and the graph name of the input graph.
```
python3 m5gnn_mem_est.py --root_path data_folder --is_train 0 --num_hidden 16 --num_layers 1 --graph_name fe_large
```
