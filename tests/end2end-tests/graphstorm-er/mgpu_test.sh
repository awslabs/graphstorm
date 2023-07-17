#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=4
NUM_INFO_TRAINERS=2
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_ep
echo "127.0.0.1" > ip_list.txt
cd $GS_HOME/inference_scripts/ep_infer
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

echo "**************dataset: ML edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --save-embed-path /data/gsgnn_er/emb/ --save-model-path /data/gsgnn_er/ --topk-model-to-save 1 --save-model-frequency 1000 --num-epochs 3 | tee train_log.txt

error_and_exit ${PIPESTATUS[0]}

# check prints
cnt=$(grep "save_embed_path: /data/gsgnn_er/emb/" train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_embed_path"
    exit -1
fi

cnt=$(grep "save_model_path: /data/gsgnn_er/" train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_model_path"
    exit -1
fi

bst_cnt=$(grep "Best Test rmse" train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test rmse"
    exit -1
fi

cnt=$(grep "| Test rmse" train_log.txt | wc -l)
if test $cnt -lt $bst_cnt
then
    echo "We use SageMaker task tracker, we should have Test rmse"
    exit -1
fi

bst_cnt=$(grep "Best Validation rmse" train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation rmse"
    exit -1
fi

cnt=$(grep "Validation rmse" train_log.txt | wc -l)
if test $cnt -lt $bst_cnt
then
    echo "We use SageMaker task tracker, we should have Validation rmse"
    exit -1
fi

cnt=$(grep "Best Iteration" train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Iteration"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_er/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi

best_epoch=$(grep "successfully save the model to" train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch"

echo "**************dataset: ML edge regression, do inference on saved model"
python3 -m graphstorm.run.gs_edge_regression --inference --workspace $GS_HOME/inference_scripts/ep_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er_infer.yaml --use-mini-batch-infer false --save-embed-path /data/gsgnn_er/infer-emb/ --restore-model-path /data/gsgnn_er/epoch-$best_epoch/ | tee log.txt

error_and_exit ${PIPESTATUS[0]}

cnt=$(grep "| Test rmse" log.txt | wc -l)
if test $cnt -ne 1
then
    echo "We do test, should have test rmse"
    exit -1
fi

bst_cnt=$(grep "Best Test rmse" log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test rmse"
    exit -1
fi

bst_cnt=$(grep "Best Validation rmse" log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation rmse"
    exit -1
fi

cnt=$(grep "Validation rmse" log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Validation rmse"
    exit -1
fi

cnt=$(grep "Best Iteration" log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Iteration"
    exit -1
fi

cd $GS_HOME/tests/end2end-tests/
python3 check_infer.py --train_embout /data/gsgnn_er/emb/ --infer_embout /data/gsgnn_er/infer-emb/

error_and_exit $?

echo "**************dataset: ML edge regression, do inference on saved model without test mask in the graph"
python3 -m graphstorm.run.gs_edge_regression --inference --workspace $GS_HOME/inference_scripts/ep_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_infer_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er_infer.yaml --use-mini-batch-infer false --save-embed-path /data/gsgnn_er/infer-emb/ --restore-model-path /data/gsgnn_er/epoch-$best_epoch/ --no-validation true

error_and_exit $?

rm -fr /data/gsgnn_er/*

echo "**************dataset: ML edge regression, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, edge decoder feat: rate"
python3 -m graphstorm.run.gs_edge_regression --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_er.yaml --save-embed-path /data/gsgnn_er/emb/ --save-model-path /data/gsgnn_er/ --topk-model-to-save 1 --save-model-frequency 1000 --num-epochs 3 --decoder-edge-feat user,rating,movie:rate --decoder-type MLPEFeatEdgeDecoder

error_and_exit $?
rm -fr /data/gsgnn_er/*
