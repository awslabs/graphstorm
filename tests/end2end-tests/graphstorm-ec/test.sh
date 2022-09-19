#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=1
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_ec

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
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec_huggingface.py --cf test_ec.yaml --train-nodes 0 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json"

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, eval_metric: precision_recall"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec_huggingface.py --cf test_ec.yaml --train-nodes 0 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json --eval-metric precision_recall"

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, eval_metric: precision_recall accuracy"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec_huggingface.py --cf test_ec.yaml --train-nodes 0 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json --eval-metric precision_recall accuracy"

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, eval_metric: roc_auc"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec_huggingface.py --cf test_ec.yaml --train-nodes 0 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json --eval-metric roc_auc"

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec_huggingface.py --cf test_ec.yaml --train-nodes 0 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json --mini-batch-infer false"

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: finetune HF BERT, BERT nodes: movie, inference: mini-batch"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec_huggingface.py --cf test_ec.yaml --train-nodes 10 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json"

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, remove-target-edge: false"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec_huggingface.py --cf test_ec.yaml --train-nodes 0 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json --remove-target-edge false"

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, remove-target-edge: true"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec_huggingface.py --cf test_ec.yaml --train-nodes 0 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json --remove-target-edge true"

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 2, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec_huggingface.py --cf test_ec.yaml --train-nodes 0 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json --fanout '10,15' --n-layers 2"

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 2, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, fanout: different per etype, eval_fanout: different per etype"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec_huggingface.py --cf test_ec.yaml --train-nodes 0 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json --fanout 'r0:10@r1:2@r0-rev:10@r1-rev:2,r0:10@r1:0@r0-rev:10@r1-rev:0' --eval-fanout 'r0:10@r1:2@r0-rev:10@r1-rev:2,r0:10@r1:0@r0-rev:10@r1-rev:0' --n-layers 2"

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, exclude-training-targets: True"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_undirected_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec_huggingface.py --cf test_ec.yaml --train-nodes 0 --num-gpus 1 --part-config /data/test_ec_undirected_1p_4t/test.json --exclude-training-targets True --reverse-edge-types-map node,r0,rev-r0,item"

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, exclude-training-targets: True, custom features"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_nodefeat_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec_huggingface.py --cf test_ec.yaml --train-nodes 0 --num-gpus 1 --part-config /data/test_ec_nodefeat_1p_4t/test.json --exclude-training-targets True --reverse-edge-types-map node,r0,rev-r0,item --feat-name embedding"

error_and_exit $?

echo "**************dataset: Test edge classification, RGAT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec_huggingface.py --cf test_ec.yaml --model-encoder-type rgat --train-nodes 0 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json"

error_and_exit $?


echo "**************dataset: Generated multilabel EC test, RGCN layer: 1, node feat: generated feature, inference: mini-batch, exclude-training-targets: True"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_multilabel_ec_1p_4t/multilabel-ec-test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_pure_gnn_ec.py --cf test_ec.yaml --graph-name multilabel-ec-test --num-gpus 1 --part-config /data/test_multilabel_ec_1p_4t/multilabel-ec-test.json --exclude-training-targets True --reverse-edge-types-map ntype0,r1,rev-r1,ntype1 --label-field label --target-etype ntype0,r1,ntype1 --multilabel true --num-classes 6 --feat-name feat"

error_and_exit $?

echo "**************dataset: Generated multilabel EC test, RGCN layer: 1, node feat: generated feature, inference: full graph, exclude-training-targets: True"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_multilabel_ec_1p_4t/multilabel-ec-test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_pure_gnn_ec.py --cf test_ec.yaml --graph-name multilabel-ec-test --num-gpus 1 --part-config /data/test_multilabel_ec_1p_4t/multilabel-ec-test.json --exclude-training-targets True --reverse-edge-types-map ntype0,r1,rev-r1,ntype1 --label-field label --target-etype ntype0,r1,ntype1 --multilabel true --num-classes 6 --feat-name feat --mini-batch-infer false"

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, imbalance-class"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec_huggingface.py --cf test_ec.yaml --train-nodes 0 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json --mini-batch-infer false --imbalance-class-weights 1,1,1,1,2,1,1,1,1,2,1"

error_and_exit $?

echo "**************dataset: Test edge classification, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch early stop"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_ec_1p_4t/test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec_huggingface.py --cf test_ec.yaml --train-nodes 0 --num-gpus 1 --part-config /data/test_ec_1p_4t/test.json --enable-early-stop True --call-to-consider-early-stop 2 -e 20 --window-for-early-stop 5" | tee exec.log

error_and_exit $?

# check early stop
cnt=$(cat exec.log | grep "Evaluation step" | wc -l)
if test $cnt -eq 20
then
	echo "Early stop should work, but it didn't"
	exit -1
fi


if test $cnt -le 6
then
	echo "Need at least 7 iters"
	exit -1
fi


echo 'Done'
