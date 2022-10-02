#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=1
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/language_model_ec

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


echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/language_model_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 lm_ec_huggingface.py --cf test_ec.yaml --train-nodes 0 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json"

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: finetune HF BERT, BERT nodes: movie, inference: mini-batch"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/language_model_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 lm_ec_huggingface.py --cf test_ec.yaml --train-nodes 10 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json"

error_and_exit $?

echo "**************dataset: Test edge classification, full nodes BERT, node feat: finetune HF BERT, BERT nodes: movie, inference: mini-batch, decoder-type: DenseBiDecoder"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/language_model_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 lm_ec_huggingface.py --cf test_ec.yaml --train-nodes -1 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json --decoder-type DenseBiDecoder --batch-size 10" | tee log.txt

error_and_exit $?

echo 'Done'
