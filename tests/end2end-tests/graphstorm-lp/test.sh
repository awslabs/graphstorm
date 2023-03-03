#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=1
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_lp

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

echo "Test GraphStorm link prediction"

date

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, negative_sampler: joint, exclude_training_targets: false"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --n-epochs 1 --evaluation-frequency 300"

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, negative_sampler: joint, exclude_training_targets: false, no train mask"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_no_edata_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --num-gpus $NUM_TRAINERS --part-config /data/movielen_no_edata_100k_train_val_1p_4t/movie-lens-100k.json --n-epochs 1 --evaluation-frequency 300 --no-validation true"

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, negative_sampler: uniform, exclude_training_targets: true"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --negative-sampler uniform --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --n-epochs 1 --evaluation-frequency 300"

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, exclude_training_targets: false"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --use-node-embeddings true --mini-batch-infer false --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --n-epochs 1 --evaluation-frequency 300"

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, negative_sampler: joint, exclude_training_targets: false"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --fanout '10,15' --n-layers 2 --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --n-epochs 1 --evaluation-frequency 300"

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, exclude_training_targets: false"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --n-epochs 1 --evaluation-frequency 300"

error_and_exit $?

date

echo 'Done'
