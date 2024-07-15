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
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --num-epochs 1

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, no test"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_no_test_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml  --num-epochs 1 --logging-file /tmp/train_log.txt

error_and_exit $?

bst_cnt=$(grep "Best Test rmse: N/A" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "Test set is empty we should have Best Test rmse: N/A"
    exit -1
fi

rm /tmp/train_log.txt

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, eval_metric: mse"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --eval-metric mse --num-epochs 1 --decoder-norm batch

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, eval_metric: mae"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --eval-metric mae --num-epochs 1 --decoder-norm layer

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --use-mini-batch-infer false --num-epochs 1

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, save model and emb"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --use-mini-batch-infer false --save-model-path ./model/er_model/ --topk-model-to-save 3 --save-embed-path ./model/ml-emb/ --num-epochs 1 --save-model-frequency 1000

error_and_exit $?

echo "**************dataset: Test edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, Backend nccl"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --num-epochs 1 --node-feat-name movie:title user:feat --backend nccl

error_and_exit $?

date

echo 'Done'
