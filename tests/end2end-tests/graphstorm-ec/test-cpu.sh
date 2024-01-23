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

echo "Test GraphStorm edge classification"

date

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --num-epochs 1

error_and_exit $?

# TODO(zhengda) Failure found during evaluation of the auc metric returning -1 multiclass format is not supported
# In Jan. 24, change all behavior of evaluation errors as stop code running, rather than returning -1. So change this
# test's eval_metric from "precision_recall" to "roc_auc". The movielens edge classification is a mutltiple class
# task, which sklearn's precision_recall is not designed for.
echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, eval_metric: roc_auc"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222  --cf ml_ec.yaml --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --eval-metric roc_auc --num-epochs 1

error_and_exit $?

echo "**************dataset: Generated multilabel EC test, RGCN layer: 1, node feat: generated feature, inference: mini-batch, exclude-training-targets: True"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222  --cf ml_ec.yaml --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --multilabel true --node-feat-name movie:title --num-epochs 1

error_and_exit $?

date

echo 'Done'
