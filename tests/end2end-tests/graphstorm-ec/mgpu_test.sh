#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=4
NUM_INFO_TRAINERS=2
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/m5gnn_ec
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

echo "**************dataset: Generated multilabel EC test, RGCN layer: 1, node feat: generated feature, inference: full graph, exclude-training-targets: True, save model"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/m5gnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_multilabel_ec_1p_4t/multilabel-ec-test.json --ip_config ip_list.txt --ssh_port 2222 "python3 m5gnn_pure_gnn_ec.py --cf test_ec.yaml --graph-name multilabel-ec-test --num-gpus $NUM_TRAINERS --part-config /data/test_multilabel_ec_1p_4t/multilabel-ec-test.json --exclude-training-targets True --reverse-edge-types-map ntype0,r1,rev-r1,ntype1 --label-field label --target-etype ntype0,r1,ntype1 --multilabel true --num-classes 6 --feat-name feat --mini-batch-infer false --save-embeds-path /data/m5gnn_ec/emb/ --save-model-path /data/m5gnn_ec/ --save-model-per-iter 0"

error_and_exit $?

echo "**************dataset: Generated multilabel EC test, do inference on saved model"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/ep_infer --num_trainers $NUM_INFO_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_multilabel_ec_1p_4t/multilabel-ec-test.json --ip_config ip_list.txt --ssh_port 2222 "python3 ep_infer_huggingface.py --cf test_ec_infer.yaml --graph-name multilabel-ec-test --num-gpus $NUM_INFO_TRAINERS --part-config /data/test_multilabel_ec_1p_4t/multilabel-ec-test.json --label-field label --target-etype ntype0,r1,ntype1 --multilabel true --num-classes 6 --feat-name feat --mini-batch-infer false --save-embeds-path /data/m5gnn_ec/infer-emb/ --restore-model-path /data/m5gnn_ec/-2/" | tee log.txt

error_and_exit $?

cnt=$(grep "Test accuracy" log.txt | wc -l)
if test $cnt -ne 1
then
    echo "We do test, should have test accuracy"
    exit -1
fi

cd $GS_HOME/tests/end2end-tests/
python3 check_infer.py --train_embout /data/m5gnn_ec/emb/-2/ --infer_embout /data/m5gnn_ec/infer-emb/ --edge_prediction
