#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=1
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_er

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

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_er/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_er_huggingface.py --cf ml_er.yaml --num-gpus 1 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --n-epochs 1"

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, eval_metric: mse"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_er/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_er_huggingface.py --cf ml_er.yaml --num-gpus 1 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --eval-metric mse --n-epochs 1"

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_er/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_er_huggingface.py --cf ml_er.yaml --num-gpus 1 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --mini-batch-infer false --n-epochs 1"

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, save model and emb"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_er/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_er_huggingface.py --cf ml_er.yaml --num-gpus 1 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --mini-batch-infer false --save-model-path ./model/er_model/ --topk-model-to-save 3 --save-embeds-path ./model/ml-emb/ --n-epochs 1"

error_and_exit $?
