#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=4
NUM_INFERs=2
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_nc
echo "127.0.0.1" > ip_list.txt
cd $GS_HOME/inference_scripts/np_infer
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

echo "**************dataset: MovieLens classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch save model save emb node"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_nc_huggingface.py --cf ml_nc.yaml --train-nodes 0 --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --save-model-path /data/gsgnn_nc_ml/ --save-embeds-path /data/gsgnn_nc_ml/emb/ --n-epochs 3"

error_and_exit $?

echo "**************dataset: Movielens, do inference on saved model, decoder: dot"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/np_infer/ --num_trainers $NUM_INFERs --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 np_infer_huggingface.py --cf ml_nc_infer.yaml --mini-batch-infer false --num-gpus $NUM_INFERs --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --save-embeds-path /data/gsgnn_nc_ml/infer-emb/ --restore-model-path /data/gsgnn_nc_ml/-2/ --save-predict-path /data/gsgnn_nc_ml/prediction/ --early-stop-strategy consecutive_increase" | tee log.txt

error_and_exit $?

cnt=$(grep "test accuracy" log.txt | wc -l)
if test $cnt -ne 1
then
    echo "We do test, should have mrr"
    exit -1
fi

cd $GS_HOME/tests/end2end-tests/graphstorm-nc/
python3 $GS_HOME/tests/end2end-tests/check_infer.py --train_embout /data/gsgnn_nc_ml/emb/-2/ --infer_embout /data/gsgnn_nc_ml/infer-emb/

cnt=$(ls /data/gsgnn_nc_ml/prediction/ | grep predict.pt | wc -l)
if test $cnt -ne 1
then
    echo "DistMult inference outputs edge embedding"
    exit -1
fi


echo "**************dataset: MovieLens classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch save model save emb node, early stop"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_nc_huggingface.py --cf ml_nc.yaml --train-nodes 0 --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --save-model-path /data/gsgnn_nc_ml/ --save-embeds-path /data/gsgnn_nc_ml/emb/ --enable-early-stop True --call-to-consider-early-stop 2 -e 20 --window-for-early-stop 3" | tee exec.log

error_and_exit $?

# check early stop
cnt=$(cat exec.log | grep "Evaluation step" | wc -l)
if test $cnt -eq 20
then
	echo "Early stop should work, but it didn't"
	exit -1
fi

if test $cnt -le 4
then
	echo "Need at least 5 iters"
	exit -1
fi
