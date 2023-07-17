#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=1
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_ep

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

echo "Test GraphStorm edge regression"

date

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --num-epochs 1

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, eval_metric: mse"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --eval-metric mse --num-epochs 1

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, eval_metric: mae"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --eval-metric mae --num-epochs 1

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --use-mini-batch-infer false --num-epochs 1

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, save model and emb"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --use-mini-batch-infer false --save-model-path ./model/er_model/ --topk-model-to-save 3 --save-embed-path ./model/ml-emb/ --num-epochs 1 --save-model-frequency 1000

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, ngnn layer between GNN layer: 1"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --use-mini-batch-infer false --num-epochs 1 --num-gnn-ngnn-layer 1 --save-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model

error_and_exit $?

python3 -m graphstorm.run.gs_edge_regression --inference --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --use-mini-batch-infer false --restore-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model/epoch-0/

error_and_exit $?

rm -R ./models/movielen_100k/train_val/movielen_100k_ngnn_model

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, ngnn layer in input layer: 1"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --use-mini-batch-infer false --num-epochs 1 --num-input-ngnn-layers 1 --save-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model

error_and_exit $?

python3 -m graphstorm.run.gs_edge_regression --inference --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --use-mini-batch-infer false --restore-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model/epoch-0/

error_and_exit $?

rm -R ./models/movielen_100k/train_val/movielen_100k_ngnn_model

date

echo 'Done'
