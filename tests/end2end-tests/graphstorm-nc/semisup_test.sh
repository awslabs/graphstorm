#!/bin/bash

cd /develop/graphstorm
DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=4
NUM_INFERs=2
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_np
echo "127.0.0.1" > ip_list.txt

echo "**************dataset: MovieLens classification, GLEM co-training, RGCN layer: 1, node feat: BERT nodes: movie, user inference: mini-batch save model save emb node"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_text_train_val_1p_4t/movie-lens-100k-text.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_utext_glem.yml  --save-model-path /data/gsgnn_nc_ml_text/ --topk-model-to-save 1 --num-epochs 3 | tee train_log.txt


## debug data and loader
python3 -m torch.distributed.launch -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 1 --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_text_train_val_1p_4t/movie-lens-100k-text.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_utext_glem.yml  --save-model-path /data/gsgnn_nc_ml_text/ --topk-model-to-save 1 --num-epochs 3


## Set up public dataset 
# (are there ogb datasets with unlabeled nodes)?
cd /develop/graphstorm

python3 tools/partition_graph.py \
    --dataset ogbn-papers100M \
    --filepath /data/dataset/movie-lens-100k-text/ \
    --num-parts 1 \
    --num-trainers-per-machine 8 \
    --output /data/movie-lens-100k-text_1p_8t


python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 1 --num-servers 1 --num-samplers 0 --part-config /data/ogbn-papers100M-4p/ogbn-papers100M.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_utext_glem.yml --topk-model-to-save 1 --num-epochs 3

