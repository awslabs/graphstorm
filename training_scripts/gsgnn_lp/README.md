# ARXIV Link Prediction Example
Arxiv link prediction example serves as the simplest GSGNN example. It shows how to use yaml files to choose difference configurations of GSGNN.

## Preparation
You need to create a pre-processed arxiv link prediction dataset before training. Following are the example CMDs to create such a dataset:

```
$ GS_HOME=/fsx-dev/xiangsx/home/workspace/graph-storm
$ export PYTHONPATH=$GS_HOME/python/
$ cd $GS_HOME/training_scripts/gsgnn_lp
$ aws s3 cp --recursive s3://graphstorm-example/arxiv/ogbn-arxiv-raw/ ogbn-arxiv-raw/
$ python3 python3 -u $GS_HOME/tools/partition_graph_lp.py --dataset ogbn-arxiv --filepath ogbn-arxiv-raw --num_parts 2 --num_trainers_per_machine 4 --output ogb_arxiv_train_val_2p_4t
```

The output file is ogb_arxiv_train_val_1p_4t/. It contains a partitioned DGLGraph with a signle partition.

## Training
After copying the ogb_arxiv_train_val_1p_4t folder into current location (under gsgnn_lp), We can launch the training task.

```
$ DGL_HOME=/fsx-dev/xiangsx/home/workspace/dgl/dgl
$ python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_lp \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config ogb_arxiv_train_val_1p_4t/ogbn-arxiv.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
    --ip_config ip_list.txt \
    "python3 gsgnn_lp_huggingface.py --cf arxiv_lp_hf.yaml"
```

## Difference configurations
train+validation+mixed-precision-O2+joint-sampler+save-model+save-embeds
```
python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_lp \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config ogb_arxiv_train_val_1p_4t/ogbn-arxiv.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
     --ip_config ip_list.txt \
     "python3 gsgnn_lp_huggingface.py --cf arxiv_lp_hf.yaml"
```

train+validation+mixed-precision-O1+local-uniform
```
python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_lp \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config ogb_arxiv_train_val_1p_4t/ogbn-arxiv.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
     --ip_config ip_list.txt \
     "python3 gsgnn_lp_huggingface.py --cf arxiv_lp_hf.yaml --mp-opt-level O1 --save-model-path none --save-embeds-path none --negative-sampler uniform"
```

train+validation+mixed-precision-O1+joint+full-graph-infer
```
python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_lp \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config ogb_arxiv_train_val_1p_4t/ogbn-arxiv.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
     --ip_config ip_list.txt \
     "python3 gsgnn_lp_huggingface.py --cf arxiv_lp_hf.yaml --mp-opt-level O1 --save-model-path none --save-embeds-path none --save-model-per-iters 0 --mini-batch-infer false"
```

train-only+mixed-precision-02+joint-sampler+save-model
```
python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_lp \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config ogb_arxiv_train_val_1p_4t/ogbn-arxiv.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
    --ip_config ip_list.txt \
    "python3 gsgnn_lp_huggingface.py --cf arxiv_lp_hf.yaml --part-config 'ogb_arxiv_train_1p_4t/ogbn-arxiv.json' save-model-path './models/ogb_arxiv/train_only/ogb_arxiv_train_1p_4t_model' --save-embeds-path none --batch-size 64"
```

train+validation+localuniform-sampler+bert-cache
```
python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_lp \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config ogb_arxiv_train_val_1p_4t/ogbn-arxiv.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
     --ip_config ip_list.txt \
     "python3 gsgnn_lp_huggingface.py --cf arxiv_lp_hf.yaml --use-bert-cache true --refresh-cache true --mixed-precision false --save-model-path none --save-embeds-path none --negative-sampler localuniform"
```

train+validation+mixed-precision-O2+joint-sampler+save-model+save-embeds+user-node-embedding
```
python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_lp \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config ogb_arxiv_train_val_1p_4t/ogbn-arxiv.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
    --ip_config ip_list.txt
    "python3 gsgnn_lp_huggingface.py --cf arxiv_lp_hf.yaml --use-node-embeddings true"
```

## None-Bert Training
Generate a graph data without g.nodes['node'].data['text_idx']
```
$ python3 $GS_HOME/tools/gen_ogbn_dataset.py --filepath ogbn-arxiv-raw/ --savepath ogb-arxiv-origin/ --edge_pct 0.8 --retain_original_features True
$ python3 -u $GS_HOME/tools/partition_graph_lp.py --dataset ogbn-arxiv --filepath ogb-arxiv-origin/ --num_parts 1 --num_trainers_per_machine 4 --output ogb_arxiv_origin_1p_4t
```

```
$ DGL_HOME=/fsx-dev/xiangsx/home/workspace/dgl/dgl
$ python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_lp \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config ogb_arxiv_origin_1p_4t/ogbn-arxiv.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
    --ip_config ip_list.txt \
    "python3 gsgnn_pure_gnn_lp.py --cf gsgnn_pure_gnn_lp.yaml"
```

## Movielens link prediction

Preparing movielens dataset for link prediction. Movie nodes use movie title as text feature and user nodes are featureless.
```
$ GS_HOME=/fsx-dev/xiangsx/home/workspace/graph-storm
$ export PYTHONPATH=$GS_HOME/python/
$ wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
$ unzip ml-100k.zip
$ rm ml-100k.zip
$ python3 $GS_HOME/python/graphstorm/data/tools/preprocess_movielens.py \
    --input_path ml-100k --output_path movielen-data
$ rm -R ml-100k
$ python3 /graphstorm/tools/construct_graph.py --name movie-lens-100k\
	--undirected \
    --filepath movielen-data \
    --output data \
    --dist_output movielen_100k_train_val_1p_4t \
    --num_dataset_workers 10 \
    --hf_bert_model bert-base-uncased \
    --ntext_fields "movie:title" \
    --num_parts 1 \
    --num_trainers_per_machine 4 \
    --balance_train \
    --balance_edges \
    --generate_new_split true \
    --compute_bert_emb true \
    --device 0 \
    --remove_text_tokens true
```

Training
```
DGL_HOME=/fsx-dev/xiangsx/home/workspace/dgl/dgl
python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_lp \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config movielen_100k_train_val_1p_4t/movie-lens-100k.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
    --ip_config ip_list.txt \
    "python3 gsgnn_lp_huggingface.py --cf ml_lp.yaml"
```
