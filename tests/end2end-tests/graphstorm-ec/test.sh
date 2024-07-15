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

echo "**************standalone"
python3 $GS_HOME/python/graphstorm/run/gsgnn_ep/gsgnn_ep.py --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --cf $GS_HOME/training_scripts/gsgnn_ep/ml_ec.yaml

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --num-epochs 1

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT & construct, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --num-epochs 1 --node-feat-name movie:title --construct-feat-ntype user

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, no test"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ec_no_test_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --num-epochs 1 --logging-file /tmp/train_log.txt

error_and_exit $?

bst_cnt=$(grep "Best Test accuracy: N/A" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "Test set is empty we should have Best Test accuracy: N/A"
    exit -1
fi

rm /tmp/train_log.txt

mkdir -p /tmp/ML_ec_profile

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, with profiling"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --num-epochs 1 --profile-path /tmp/ML_ec_profile

error_and_exit $?

cnt=$(ls /tmp/ML_ec_profile/*.csv | wc -l)
if test $cnt -lt 1
then
    echo "Cannot find the profiling files."
    exit -1
fi

rm -R /tmp/ML_ec_profile

# TODO(zhengda) Failure found during evaluation of the auc metric returning -1 multiclass format is not supported
# 01/20/2024: (James) change all behavior of evaluation errors as broken, rather than returning -1. So change this
# test's eval_metric from "precision_recall" to "roc_auc". The movielens edge classification is a mutltiple class
# task, which sklearn's precision_recall is not designed for.
echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, eval_metric: roc_auc"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222  --cf ml_ec.yaml --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --eval-metric roc_auc --num-epochs 1 --node-feat-name movie:title

error_and_exit $?

# In Jan. 2024, change all behavior of evaluation errors as broken, rather than returning -1. So change this
# test's eval_metric from "precision_recall" to "roc_auc". The movielens edge classification is a mutltiple class
# task, which sklearn's precision_recall is not designed for.
echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, eval_metric: roc_auc"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222  --cf ml_ec.yaml --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --eval-metric roc_auc --num-epochs 1 --decoder-norm layer

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, eval_metric: roc_auc"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222  --cf ml_ec.yaml --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --eval-metric roc_auc --num-epochs 1 --decoder-norm batch
error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222  --cf ml_ec.yaml --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --use-mini-batch-infer false --num-epochs 1

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, remove-target-edge: false"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222  --cf ml_ec.yaml --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --remove-target-edge false --num-epochs 1

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 2, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, fanout: different per etype, eval_fanout: different per etype"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222  --cf ml_ec.yaml --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --fanout 'rating:10@rating-rev:2,rating:5@rating-rev:0' --eval-fanout 'rating:10@rating-rev:2,rating:5@rating-rev:0' --num-layers 2 --num-epochs 1

error_and_exit $?

echo "**************dataset: Generated multilabel EC test, RGCN layer: 1, node feat: generated feature, inference: mini-batch, exclude-training-targets: True"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222  --cf ml_ec.yaml --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --multilabel true --node-feat-name movie:title --num-epochs 1

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, imbalance-class"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222  --cf ml_ec.yaml --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --use-mini-batch-infer false --imbalance-class-weights 1,1,2,1,2 --num-epochs 1

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch early stop"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222  --cf ml_ec.yaml --part-config /data/movielen_100k_ec_1p_4t/movie-lens-100k.json --use-early-stop True --early-stop-burnin-rounds 2 -e 30 --early-stop-rounds 3 --eval-frequency 100 --lr 0.01 --logging-file /tmp/exec.log

error_and_exit $?

# check early stop
cnt=$(cat /tmp/exec.log | grep "Evaluation step" | wc -l)
if test $cnt -eq 30
then
	echo "Early stop should work, but it didn't"
	exit -1
fi


if test $cnt -le 4
then
	echo "Need at least 5 iters"
	exit -1
fi

rm /tmp/exec.log

date

echo 'Done'
