#!/bin/bash

service ssh restart

GS_HOME=$(pwd)
NUM_TRAINERS=4
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_ep

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

# Test the DistDGL graph format.
echo "********* Test the DistDGL graph format ********"
python3 $GS_HOME/tests/end2end-tests/data_process/process_movielens.py
python3 -m graphstorm.gconstruct.construct_graph --conf-file $GS_HOME/tests/end2end-tests/data_process/movielens.json --num-processes 1 --output-dir /tmp/movielens --graph-name ml --add-reverse-edges --part-method random

error_and_exit $?

echo "********* Test the DistDGL graph format with BERT embeddings ********"
python3 -m graphstorm.gconstruct.construct_graph --conf-file $GS_HOME/tests/end2end-tests/data_process/movielens_bert.json --num-processes 1 --output-dir /tmp/movielens_bert --graph-name ml --add-reverse-edges

error_and_exit $?

# check node_id mapping is saved
if test -f /tmp/movielens_bert/node_mapping.pt -ne 0
then
	echo "/tmp/movielens_bert/node_mapping.pt must exist"
	exit -1
fi

if test -f /tmp/movielens_bert/edge_mapping.pt -ne 0
then
	echo "/tmp/movielens_bert/edge_mapping.pt must exist"
	exit -1
fi

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: BERT embeddings, inference: mini-batch"

python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /tmp/movielens_bert/ml.json --ip_config ip_list.txt --ssh_port 2222 --cf ml_ec.yaml --num-epochs 1 --node-feat-name user:feat movie:title --save-embed-path /tmp/movielens_bert/emb/

error_and_exit $?

# check node embeddings are saved
ls /tmp/movielens_bert/emb/ | wc -l
cnt=$(ls /tmp/movielens_bert/emb/ | wc -l)
if test $cnt -lt 2
then
	echo "Must have node embeddings for movie and user."
	exit -1
fi

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: HF BERT, BERT nodes: movie, inference: mini-batch"

python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /tmp/movielens/ml.json --ip_config ip_list.txt --ssh_port 2222 --cf $GS_HOME/tests/end2end-tests/data_process/ml_ec_text.yaml --num-epochs 1

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /tmp/movielens/ml.json --ip_config ip_list.txt --ssh_port 2222 --cf ml_ec.yaml --num-epochs 1

error_and_exit $?

rm -R /tmp/movielens
rm -R /tmp/movielens_bert
