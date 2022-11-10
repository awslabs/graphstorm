#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=1
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/language_model_nc

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

echo "Test language model node classification"

date

echo "**************dataset: MovieLens, language model, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/language_model_nc/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 lm_nc_huggingface.py --cf lm_ml_nc_hf.yaml --train-nodes 0 --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json"

error_and_exit $?


echo "**************dataset: MovieLens, language model, node feat: fine-tune HF BERT, BERT nodes: movie, inference: mini-batch"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/language_model_nc/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 lm_nc_huggingface.py --cf lm_ml_nc_hf.yaml --train-nodes 20 --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json" | tee log.txt

error_and_exit $?

date

echo 'Done'
