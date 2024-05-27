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

error_and_exit $?

# check prints

bst_cnt=$(grep "Best Test node_classification" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test node_classification"
    exit -1
fi

cnt=$(grep "Test node_classification" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Test node_classification"
    exit -1
fi

bst_cnt=$(grep "Best Validation node_classification" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation accuracy node_classification"
    exit -1
fi

cnt=$(grep "Validation node_classification" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Validation node_classification"
    exit -1
fi

bst_cnt=$(grep "Best Test edge_classification" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test edge_classification"
    exit -1
fi

cnt=$(grep "Test edge_classification" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Test edge_classification"
    exit -1
fi

bst_cnt=$(grep "Best Validation edge_classification" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation edge_classification"
    exit -1
fi

cnt=$(grep "Validation edge_classification" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Validation edge_classification"
    exit -1
fi

bst_cnt=$(grep "Best Test edge_regression" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test edge_regression"
    exit -1
fi

cnt=$(grep "Test edge_regression" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Test edge_regression"
    exit -1
fi

bst_cnt=$(grep "Best Validation edge_regression" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation edge_regression"
    exit -1
fi

cnt=$(grep "Validation edge_regression" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Validation edge_regression"
    exit -1
fi

bst_cnt=$(grep "Best Test link_prediction" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test link_prediction"
    exit -1
fi

cnt=$(grep "Test link_prediction" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Test link_prediction"
    exit -1
fi

bst_cnt=$(grep "Best Validation link_prediction" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation link_prediction"
    exit -1
fi

cnt=$(grep "Validation link_prediction" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Validation link_prediction"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_mt/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi

echo "**************[Multi-task with learnable embedding] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, save model"
python3 -m graphstorm.run.gs_multi_task_learning -workspace $GS_HOME/training_scripts/gsgnn_mt  --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_ec_er_lp.yaml --save-model-path /data/gsgnn_mt/ --save-model-frequency 1000 --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True --use-node-embeddings True

error_and_exit $?

rm /tmp/train_log.txt
rm -frm /data/gsgnn_mt/

echo "**************[Multi-task] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, save model"
python3 -m graphstorm.run.gs_multi_task_learning -workspace $GS_HOME/training_scripts/gsgnn_mt  --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_ec_er_lp.yaml --save-model-path /data/gsgnn_mt/ --save-model-frequency 1000 --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True --use-mini-batch-infer False --save-embed-path /data/gsgnn_mt/emb/

error_and_exit $?

cnt=$(grep "save_embed_path: /data/gsgnn_mt/emb/" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_embed_path"
    exit -1
fi

echo "**************[Multi-task] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch load from saved model"
python3 -m graphstorm.run.gs_multi_task_learning -workspace $GS_HOME/training_scripts/gsgnn_mt  --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_ec_er_lp.yaml --restore-model-path /data/gsgnn_mt/epoch-2/ --save-model-path /data/gsgnn_mt_2/ --save-model-frequency 1000 --logging-file /tmp/train_log.txt --logging-level debug

error_and_exit $?

cnt=$(ls -l /data/gsgnn_mt_2/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi
