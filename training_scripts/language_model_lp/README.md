# ARXIV Node Classification Example
Arxiv node classification example serves as the simplest GSGNN example. It shows how to use yaml files to choose difference configurations of GSGNN.

## Preparation
You need to create a pre-processed arxiv node classification dataset before training. Following are the example CMDs to create such a dataset:

```
$ GS_HOME=/fsx-dev/xiangsx/home/workspace/graph-storm
$ export PYTHONPATH=$GS_HOME/python/
$ cd $GS_HOME/training_scripts/language_model_lp
$ aws s3 cp --recursive s3://search-m5-app-fsx-us-east-1-prod/FSxLustre20201016T182138Z/ivasilei/home/ogbn_text_graph_data/ogbn-arxiv/ ogbn-arxiv-raw/
$ python3 $GS_HOME/python/graphstorm/data/ogbn_datasets.py --filepath ogbn-arxiv-raw/ --savepath ogbn-arxiv/
$ python3 -u $GS_HOME/tools/partition_graph.py --dataset ogbn-arxiv --filepath ogbn-arxiv/ --num_parts 1 --num_trainers_per_machine 4 --output ogb_arxiv_lp_train_val_1p_4t
```

The output file is ogb_arxiv_lp_train_val_1p_4t/. It contains a partitioned DGLGraph with a signle partition.

## Training
After copying the ogb_arxiv_lp_train_val_1p_4t folder into current location (under arxiv_lp), We can launch the training task.

```
$ DGL_HOME=/fsx-dev/ivasilei/home/dgl/dgl
$ python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/language_model_lp --num_trainers 4 --num_servers 4 --num_samplers 0 --part_config ogb_arxiv_lp_train_val_1p_4t/ogbn-arxiv.json --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" --ip_config ip_list.txt "python3 lm_lp_huggingface.py --cf lm_lp_hf.yaml"

```
