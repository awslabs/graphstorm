#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=4
NUM_INFO_TRAINERS=2
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_mt
echo "127.0.0.1" > ip_list.txt
cd $GS_HOME/inference_scripts/lp_infer
echo "127.0.0.1" > ip_list.txt

error_and_exit () {
	# check exec status of launch.py
	status=$1
	echo $status

	if test $status -ne 0
	then
		exit -1
	fi
}

df /dev/shm -h


echo "**************[Multi-task] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, save model"
python3 -m graphstorm.run.gs_multi_task_learning -workspace $GS_HOME/training_scripts/gsgnn_mt  --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_ec_er_lp.yaml --save-model-path /data/gsgnn_mt/ --save-model-frequency 1000 --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True

echo "**************[Multi-task with learnable embedding] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, save model"
python3 -m graphstorm.run.gs_multi_task_learning -workspace $GS_HOME/training_scripts/gsgnn_mt  --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_ec_er_lp.yaml --save-model-path /data/gsgnn_mt/ --save-model-frequency 1000 --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True --use-node-embeddings True