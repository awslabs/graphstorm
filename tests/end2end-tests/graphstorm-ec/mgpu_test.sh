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

echo "**************dataset: Generated multilabel MovieLens EC, RGCN layer: 1, node feat: generated feature, inference: full graph, exclude-training-targets: True"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --exclude-training-targets True --multilabel true --num-classes 5 --node-feat-name movie:title user:feat --use-mini-batch-infer false --topk-model-to-save 1  --save-embed-path /data/gsgnn_ec/emb/ --save-model-path /data/gsgnn_ec/ --save-model-frequency 1000 --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True --backend nccl

error_and_exit $?

# check prints
cnt=$(grep "save_embed_path: /data/gsgnn_ec/emb/" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_embed_path"
    exit -1
fi

cnt=$(grep "save_model_path: /data/gsgnn_ec/" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_model_path"
    exit -1
fi

bst_cnt=$(grep "Best Test accuracy" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test accuracy"
    exit -1
fi

bst_cnt=$(grep "Best Validation accuracy" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation accuracy"
    exit -1
fi

cnt=$(grep "Best Iteration" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Iteration"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_ec/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi

best_epoch=$(grep "successfully save the model to" /tmp/train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch"

rm /tmp/train_log.txt

echo "**************dataset: Generated multilabel MovieLens EC, load only embed layer and GNN layer of the saved model to retrain"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --exclude-training-targets True --multilabel true --num-classes 5 --node-feat-name movie:title user:feat --use-mini-batch-infer false --topk-model-to-save 1  --restore-model-path /data/gsgnn_ec/epoch-$best_epoch/ --restore-model-layers embed,gnn --save-model-path /data/gsgnn_ec_retrain/ --save-model-frequency 1000

error_and_exit $?

cnt=$(ls -l /data/gsgnn_ec_retrain/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi

echo "**************dataset: Generated multilabel MovieLens EC, do inference on saved model"
python3 -m graphstorm.run.gs_edge_classification --inference --workspace $GS_HOME/inference_scripts/ep_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec_infer.yaml  --multilabel true --num-classes 5 --node-feat-name movie:title user:feat --use-mini-batch-infer false --save-embed-path /data/gsgnn_ec/infer-emb/ --restore-model-path /data/gsgnn_ec/epoch-$best_epoch/ --save-prediction-path /data/gsgnn_ec/prediction/ --logging-file /tmp/log.txt  --logging-level debug --preserve-input True --backend nccl

error_and_exit $?

cnt=$(grep "| Test accuracy" /tmp/log.txt | wc -l)
if test $cnt -ne 1
then
    echo "We do test, should have test accuracy"
    exit -1
fi

bst_cnt=$(grep "Best Test accuracy" /tmp/log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test accuracy"
    exit -1
fi

bst_cnt=$(grep "Best Validation accuracy" /tmp/log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation accuracy"
    exit -1
fi

cnt=$(grep "Validation accuracy" /tmp/log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Validation accuracy"
    exit -1
fi

cnt=$(grep "Best Iteration" /tmp/log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Iteration"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_ec/prediction/user_rating_movie/ | grep predict | wc -l)
if test $cnt != $NUM_INFO_TRAINERS * 2
then
    echo "The number of saved prediction results $cnt is not equal to the number of inferers $NUM_INFO_TRAINERS * 2 as --preserve-input is True"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_ec/prediction/user_rating_movie/ | grep src_nids | wc -l)
if test $cnt != $NUM_INFO_TRAINERS
then
    echo "The number of saved source node ids $cnt is not equal to the number of inferers $NUM_INFO_TRAINERS"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_ec/prediction/user_rating_movie/ | grep dst_nids | wc -l)
if test $cnt != $NUM_INFO_TRAINERS
then
    echo "The number of saved dst node ids results $cnt is not equal to the number of inferers $NUM_INFO_TRAINERS"
    exit -1
fi

rm /tmp/log.txt

cd $GS_HOME/tests/end2end-tests/
python3 check_infer.py --train-embout /data/gsgnn_ec/emb/ --infer-embout /data/gsgnn_ec/infer-emb/

error_and_exit $?

python3 -m graphstorm.run.gs_edge_classification --inference --workspace $GS_HOME/inference_scripts/ep_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec_infer.yaml  --multilabel true --num-classes 5 --node-feat-name movie:title user:feat --use-mini-batch-infer false --save-embed-path /data/gsgnn_ec/infer-emb/ --restore-model-path /data/gsgnn_ec/epoch-$best_epoch/ --save-prediction-path /data/gsgnn_ec/prediction-no-sf/ --logging-file /tmp/log.txt  --logging-level debug --with-shared-fs False

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_predict_result.py --infer-prediction /data/gsgnn_ec/prediction/user_rating_movie/ --no-sfs-prediction /data/gsgnn_ec/prediction-no-sf/user_rating_movie/ --edge-prediction

error_and_exit $?

echo "**************dataset: Movielens, use gen_node_embeddings to generate embeddings on edge classification"
python3 -m graphstorm.run.gs_gen_node_embedding --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --use-mini-batch-infer false --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --exclude-training-targets True --multilabel true --num-classes 5 --node-feat-name movie:title user:feat --save-embed-path /data/gsgnn_ec/save-emb/ --restore-model-path /data/gsgnn_ec/epoch-$best_epoch/ --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_ec/emb/ --infer-embout /data/gsgnn_ec/save-emb/

error_and_exit $?

echo "**************dataset: Generated multilabel MovieLens EC, do inference on saved model without test_mask"
python3 -m graphstorm.run.gs_edge_classification --inference --workspace $GS_HOME/inference_scripts/ep_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_ec_no_test_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec_infer.yaml  --multilabel true --num-classes 5 --node-feat-name movie:title user:feat --use-mini-batch-infer false --save-embed-path /data/gsgnn_ec/infer-emb/ --restore-model-path /data/gsgnn_ec/epoch-$best_epoch/ --save-prediction-path /data/gsgnn_ec/prediction/ --no-validation true

error_and_exit $?

cnt=$(ls -l /data/gsgnn_ec/prediction/user_rating_movie/ | grep parquet | wc -l)
if test $cnt != $NUM_INFO_TRAINERS
then
    echo "The number of remapped prediction results $cnt is not equal to the number of inferers $NUM_INFO_TRAINERS"
    exit -1
fi

rm -fr /data/gsgnn_ec/*

echo "**************dataset: Generated MovieLens EC, language model only, node feat: text feature, inference: full graph, train_nodes 10"
python3 -m graphstorm.run.gs_edge_classification --lm-encoder-only --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lm_encoder_train_val_1p_4t/movie-lens-100k-text.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lm_ec.yaml --num-classes 5 --use-mini-batch-infer false --topk-model-to-save 1  --save-embed-path /data/gsgnn_ec_lm/emb/ --save-model-path /data/gsgnn_ec_lm/ --save-model-frequency 1000 --logging-file /tmp/train_log.txt --preserve-input True

error_and_exit $?

best_epoch=$(grep "successfully save the model to" /tmp/train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch"

rm /tmp/train_log.txt

echo "**************dataset: Generated MovieLens EC, node feat: text feature, do inference on saved model"
python3 -m graphstorm.run.gs_edge_classification --lm-encoder-only --inference --workspace $GS_HOME/inference_scripts/ep_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lm_encoder_train_val_1p_4t/movie-lens-100k-text.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lm_ec_infer.yaml   --num-classes 5 --use-mini-batch-infer false --save-embed-path /data/gsgnn_ec_lm/infer-emb/ --restore-model-path /data/gsgnn_ec_lm/epoch-$best_epoch/ --preserve-input True

error_and_exit $?

cd $GS_HOME/tests/end2end-tests/
python3 check_infer.py --train-embout /data/gsgnn_ec_lm/emb/ --infer-embout /data/gsgnn_ec_lm/infer-emb/

error_and_exit $?
rm -fr /data/gsgnn_ec_lm/*

echo "**************dataset: Generated multilabel MovieLens EC, RGCN layer: 1, node feat: generated feature, inference: full graph, exclude-training-targets: True, decoder edge feat: label"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --exclude-training-targets True --multilabel true --num-classes 5 --node-feat-name movie:title user:feat --use-mini-batch-infer false --topk-model-to-save 1  --save-embed-path /data/gsgnn_ec/emb/ --save-model-path /data/gsgnn_ec/ --save-model-frequency 1000 --decoder-edge-feat user,rating,movie:rate --fanout 'user/rating/movie:4@movie/rating-rev/user:5,user/rating/movie:2@movie/rating-rev/user:2' --num-layers 2 --decoder-type MLPEFeatEdgeDecoder

error_and_exit $?
rm -fr /data/gsgnn_ec/*

echo "**************dataset: Generated multilabel MovieLens EC, RGCN layer: 1, node feat: generated feature, inference: full graph, exclude-training-targets: True,  Backend nccl"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --exclude-training-targets True --multilabel true --num-classes 5 --node-feat-name movie:title user:feat --use-mini-batch-infer false --num-epochs 1 --backend nccl

error_and_exit $?

echo "**************dataset: Generated multilabel MovieLens EC, RGCN layer: 1, node feat: generated feature, inference: minibatch, exclude-training-targets: True, decoder edge feat: label"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --exclude-training-targets True --multilabel true --num-classes 5 --node-feat-name movie:title user:feat --use-mini-batch-infer true --topk-model-to-save 1  --save-embed-path /data/gsgnn_ec/emb/ --save-model-path /data/gsgnn_ec/ --save-model-frequency 1000 --decoder-edge-feat user,rating,movie:rate --fanout 'user/rating/movie:4@movie/rating-rev/user:5,user/rating/movie:2@movie/rating-rev/user:2' --num-layers 2 --decoder-type MLPEFeatEdgeDecoder

error_and_exit $?
rm -fr /data/gsgnn_ec/*

echo "**************dataset: Generated multilabel MovieLens EC, RGCN layer: 1, node feat: generated feature, inference: full graph, exclude-training-targets: True, wholegraph learnable emb"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --exclude-training-targets True --use-node-embeddings true --multilabel true --num-classes 5  --use-mini-batch-infer false --topk-model-to-save 1  --save-embed-path /data/gsgnn_wg_ec/emb/ --save-model-path /data/gsgnn_wg_ec/ --save-model-frequency 1000 --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True --backend nccl --use-wholegraph-embed True

error_and_exit $?

# check prints
cnt=$(grep "save_embed_path: /data/gsgnn_wg_ec/emb/" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_embed_path"
    exit -1
fi

cnt=$(grep "save_model_path: /data/gsgnn_wg_ec/" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_model_path"
    exit -1
fi

bst_cnt=$(grep "Best Test accuracy" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test accuracy"
    exit -1
fi


best_epoch=$(grep "successfully save the model to" /tmp/train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch"

rm /tmp/train_log.txt

echo "**************dataset: Generated multilabel MovieLens EC, do inference on saved model, wholegraph learnable emb"
python3 -m graphstorm.run.gs_edge_classification --inference --workspace $GS_HOME/inference_scripts/ep_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec_infer.yaml  --multilabel true --num-classes 5 --use-node-embeddings true --use-mini-batch-infer false --save-embed-path /data/gsgnn_wg_ec/infer-emb/ --restore-model-path /data/gsgnn_wg_ec/epoch-$best_epoch/ --save-prediction-path /data/gsgnn_wg_ec/prediction/ --logging-file /tmp/log.txt  --logging-level debug --preserve-input True --backend nccl --use-wholegraph-embed True

error_and_exit $?

bst_cnt=$(grep "Best Test accuracy" /tmp/log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test accuracy"
    exit -1
fi

rm /tmp/log.txt

cd $GS_HOME/tests/end2end-tests/
python3 check_infer.py --train-embout /data/gsgnn_wg_ec/emb/ --infer-embout /data/gsgnn_wg_ec/infer-emb/

error_and_exit $?

rm -fr /data/gsgnn_wg_ec/


echo "=================== test save model and do evaluation behaviors ==================="

echo "**************dataset: Generated multilabel MovieLens EC, RGCN layer: 1, node feat: generated feature, inference: full graph, exclude-training-targets: True, no-topk save model, no eval frequency"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --exclude-training-targets True --multilabel true --num-classes 5 --node-feat-name movie:title user:feat --use-mini-batch-infer false  --save-model-path /data/gsgnn_ec/ --batch-size 64  --save-model-frequency 20 --num-epochs 1 --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True --backend nccl

error_and_exit $?

save_model_cnts=$(grep "successfully save the model to" /tmp/train_log.txt | wc -l)
do_eval_cnts=$(grep "Best Validation" /tmp/train_log.txt | wc -l)

if [ $save_model_cnts != 3 ] || [ $do_eval_cnts != 3 ]
then
    echo "The number of save models is not equal to the number of do evaluation and not equal to 3, but got $save_model_cnts and $do_eval_cnts."
    exit -1
fi

rm /tmp/train_log.txt

echo "**************dataset: Generated multilabel MovieLens EC, RGCN layer: 1, node feat: generated feature, inference: full graph, exclude-training-targets: True, no-topk save model, eval less frequently but divisible by save model frequency"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --exclude-training-targets True --multilabel true --num-classes 5 --node-feat-name movie:title user:feat --use-mini-batch-infer false  --save-model-path /data/gsgnn_ec/ --batch-size 64 --save-model-frequency 20 --eval-frequency 40 --num-epochs 1 --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True --backend nccl

error_and_exit $?

save_model_cnts=$(grep "successfully save the model to" /tmp/train_log.txt | wc -l)
do_eval_cnts=$(grep "Best Validation" /tmp/train_log.txt | wc -l)

if [ $save_model_cnts != 3 ] || [ $do_eval_cnts != 3 ]
then
    echo "The number of save models is not equal to the number of do evaluation and not equal to 3, but got $save_model_cnts and $do_eval_cnts."
    exit -1
fi

rm /tmp/train_log.txt

echo "**************dataset: Generated multilabel MovieLens EC, RGCN layer: 1, node feat: generated feature, inference: full graph, exclude-training-targets: True, no-topk save model, eval more frequently"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --exclude-training-targets True --multilabel true --num-classes 5 --node-feat-name movie:title user:feat --use-mini-batch-infer false  --save-model-path /data/gsgnn_ec/ --batch-size 64 --save-model-frequency 20 --eval-frequency 10 --num-epochs 1 --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True --backend nccl

error_and_exit $?

save_model_cnts=$(grep "successfully save the model to" /tmp/train_log.txt | wc -l)
do_eval_cnts=$(grep "Best Validation" /tmp/train_log.txt | wc -l)

if [ $save_model_cnts != 3 ]
then
    echo "The number of save models is not equal 3, but got $save_model_cnts."
    exit -1
fi

if [ $do_eval_cnts != 5 ]
then
    echo "The number of do evaluation is not equal to 5, but got $do_eval_cnts."
    exit -1
fi

rm /tmp/train_log.txt

echo "**************dataset: Generated multilabel MovieLens EC, RGCN layer: 1, node feat: generated feature, inference: full graph, exclude-training-targets: True, tiny valset"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_small_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --exclude-training-targets True --node-feat-name movie:title user:feat --use-mini-batch-infer false --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True --backend nccl

error_and_exit $?

rm -fr /tmp/*
