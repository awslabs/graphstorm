#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=4
NUM_INFO_TRAINERS=2
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_ec
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
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_multilabel_ec_1p_4t/multilabel-ec-test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_pure_gnn_ec.py --cf test_ec.yaml --graph-name multilabel-ec-test --num-gpus $NUM_TRAINERS --part-config /data/test_multilabel_ec_1p_4t/multilabel-ec-test.json --exclude-training-targets True --reverse-edge-types-map ntype0,r1,rev-r1,ntype1 --label-field label --target-etype ntype0,r1,ntype1 --multilabel true --num-classes 6 --feat-name feat --mini-batch-infer false --save-embeds-path /data/gsgnn_ec/emb/ --save-model-path /data/gsgnn_ec/ --topk-model-to-save 3 --save-model-per-iter 1000" | tee train_log.txt

error_and_exit $?

# check prints
cnt=$(grep "save_embeds_path: /data/gsgnn_ec/emb/" train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_embeds_pathy"
    exit -1
fi

cnt=$(grep "save_model_path: /data/gsgnn_ec/" train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_model_path"
    exit -1
fi

bst_cnt=$(grep "Best Test accuracy" train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test accuracy"
    exit -1
fi

cnt=$(grep "| Test accuracy" train_log.txt | wc -l)
if test $cnt -lt $(1+$bst_cnt)
then
    echo "We use SageMaker task tracker, we should have Test accuracy"
    exit -1
fi

bst_cnt=$(grep "Best Validation accuracy" train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation accuracy"
    exit -1
fi

cnt=$(grep "Validation accuracy" train_log.txt | wc -l)
if test $cnt -lt $(1+$bst_cnt)
then
    echo "We use SageMaker task tracker, we should have Validation accuracy"
    exit -1
fi

cnt=$(grep "Best Iteration" train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Iteration"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_ec/ | grep epoch | wc -l)
if test $cnt != 3
then
    echo "The number of save models $cnt is not equal to the specified topk 3"
    exit -1
fi

echo "**************dataset: Generated multilabel EC test, do inference on saved model"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/ep_infer --num_trainers $NUM_INFO_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_multilabel_ec_1p_4t/multilabel-ec-test.json --ip_config ip_list.txt --ssh_port 2222 "python3 ep_infer_huggingface.py --cf test_ec_infer.yaml --graph-name multilabel-ec-test --num-gpus $NUM_INFO_TRAINERS --part-config /data/test_multilabel_ec_1p_4t/multilabel-ec-test.json --label-field label --target-etype ntype0,r1,ntype1 --multilabel true --num-classes 6 --feat-name feat --mini-batch-infer false --save-embeds-path /data/gsgnn_ec/infer-emb/ --restore-model-path /data/gsgnn_ec/epoch-2/" | tee log.txt

error_and_exit $?

cnt=$(grep "| Test accuracy" log.txt | wc -l)
if test $cnt -ne 1
then
    echo "We do test, should have test accuracy"
    exit -1
fi

bst_cnt=$(grep "Best Test accuracy" log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test accuracy"
    exit -1
fi

bst_cnt=$(grep "Best Validation accuracy" log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation accuracy"
    exit -1
fi

cnt=$(grep "Validation accuracy" log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Validation accuracy"
    exit -1
fi

cnt=$(grep "Best Iteration" log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Iteration"
    exit -1
fi

cd $GS_HOME/tests/end2end-tests/
python3 check_infer.py --train_embout /data/gsgnn_ec/emb/epoch-2/ --infer_embout /data/gsgnn_ec/infer-emb/ --edge_prediction
