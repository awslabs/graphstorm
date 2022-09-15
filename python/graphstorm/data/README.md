# Build Ogbn-papers100m dataset
We need to download raw papers100m dataset from S3:
```
$ aws s3 cp --recursive s3://zd-test-data/paper100m-raw/ ./paper100m-raw
$ mkdir dataset
$ cd dataset
$ aws s3 cp --recursive s3://zd-test-data/ogbn_papers100M/ ./ogbn_papers100M/
$ cd ..
```

Run data processing
```
$ M5GNN_HOME=/fsx-dev/xiangsx/home/workspace/m5-gnn
$ export PYTHONPATH=$M5GNN_HOME/python/
$ python3 M5GNN_HOME/python/graphstorm/data/ogbn_datasets.py --filepath paper100m-raw --savepath ./paper100m-processed-512/ --dataset ogbn-papers100M --bert_model_name "allenai/scibert_scivocab_uncased"
```

Partition the graph
```
$ python3 M5GNN_HOME/tools/partition_graph.py --dataset ogbn-papers100m --filepath ./paper100m-processed-512/ --num_parts 8 --predict_ntypes "node" --balance_train --num_trainers_per_machine 8 --output ./ogbn_papers100m_nc_8p_8t/
```