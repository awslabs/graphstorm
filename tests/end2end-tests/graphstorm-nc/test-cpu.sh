#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=1
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_np

echo "127.0.0.1" > ip_list.txt

cat ip_list.txt

error_and_exit () {
	# check exec status of launch.py
	status=$1
	echo $status

	if test $status -ne 0
	then
		exit -1
	fi
}

echo "Test GraphStorm node classification"

date

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --use-mini-batch-infer false

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, mlp layer between GNN layer: 1"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --use-mini-batch-infer false --num-ffn-layers-in-gnn 1 --save-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model

error_and_exit $?

python3 -m graphstorm.run.gs_node_classification --inference --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --restore-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model/epoch-1/

error_and_exit $?

rm -R ./models/movielen_100k/train_val/movielen_100k_ngnn_model

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --use-node-embeddings true --use-mini-batch-infer false

error_and_exit $?

echo "**************dataset: MovieLens, RGAT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --model-encoder-type rgat

error_and_exit $?

echo "**************dataset: MovieLens, HGT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --model-encoder-type hgt

error_and_exit $?

date

echo 'Done'
