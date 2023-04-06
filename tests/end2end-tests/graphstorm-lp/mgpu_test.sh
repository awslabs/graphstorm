#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=4
NUM_INFO_TRAINERS=2
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_lp
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

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, exclude_training_targets: true, save model"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --save-model-path /data/gsgnn_lp_ml_dot/ --topk-model-to-save 1 --save-model-per-iter 1000 --save-embed-path /data/gsgnn_lp_ml_dot/emb/" | tee train_log.txt

error_and_exit ${PIPESTATUS[0]}

# check prints
cnt=$(grep "save_embed_path: /data/gsgnn_lp_ml_dot/emb/" train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_embed_path"
    exit -1
fi

cnt=$(grep "save_model_path: /data/gsgnn_lp_ml_dot/" train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_model_path"
    exit -1
fi

bst_cnt=$(grep "Best Test mrr" train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test mrr"
    exit -1
fi

cnt=$(grep "| Test mrr" train_log.txt | wc -l)
if test $cnt -lt $bst_cnt
then
    echo "We use SageMaker task tracker, we should have Test mrr"
    exit -1
fi

bst_cnt=$(grep "Best Validation mrr" train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation mrr"
    exit -1
fi

cnt=$(grep "Validation mrr" train_log.txt | wc -l)
if test $cnt -lt $bst_cnt
then
    echo "We use SageMaker task tracker, we should have Validation mrr"
    exit -1
fi

cnt=$(grep "Best Iteration" train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Iteration"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_lp_ml_dot/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi

best_epoch_dot=$(grep "successfully save the model to" train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_dot"

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, decoder: DistMult, exclude_training_targets: true, save model"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --save-model-path /data/gsgnn_lp_ml_distmult/ --topk-model-to-save 1 --save-model-per-iter 1000 --save-embed-path /data/gsgnn_lp_ml_distmult/emb/ --use-dot-product False --train-etype user,rating,movie movie,rating-rev,user" | tee train_log.txt

error_and_exit ${PIPESTATUS[0]}

cnt=$(ls -l /data/gsgnn_lp_ml_distmult/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi

best_epoch_distmult=$(grep "successfully save the model to" train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_distmult"

echo "**************dataset: Movielens, do inference on saved model, decoder: dot"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/lp_infer --num_trainers $NUM_INFO_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 lp_infer_gnn.py --cf ml_lp_infer.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --num-gpus $NUM_INFO_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --save-embed-path /data/gsgnn_lp_ml_dot/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_dot/epoch-$best_epoch_dot/" | tee log.txt

error_and_exit ${PIPESTATUS[0]}

cnt=$(grep "| Test mrr" log.txt | wc -l)
if test $cnt -ne 1
then
    echo "We do test, should have mrr"
    exit -1
fi

bst_cnt=$(grep "Best Test mrr" log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test mrr"
    exit -1
fi

bst_cnt=$(grep "Best Validation mrr" log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation mrr"
    exit -1
fi

cnt=$(grep "Validation mrr" log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Validation mrr"
    exit -1
fi

cnt=$(grep "Best Iteration" log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Iteration"
    exit -1
fi

echo "**************dataset: Movielens, do inference on saved model, decoder: DistMult"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/lp_infer --num_trainers $NUM_INFO_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 lp_infer_gnn.py --cf ml_lp_infer.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --num-gpus $NUM_INFO_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --save-embed-path /data/gsgnn_lp_ml_distmult/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_distmult/epoch-$best_epoch_distmult/ --use-dot-product False --no-validation True --train-etype user,rating,movie movie,rating-rev,user" | tee log2.txt

error_and_exit ${PIPESTATUS[0]}

cd $GS_HOME/tests/end2end-tests/graphstorm-lp/
python3 $GS_HOME/tests/end2end-tests/check_infer.py --train_embout /data/gsgnn_lp_ml_dot/emb/ --infer_embout /data/gsgnn_lp_ml_dot/infer-emb/ --link_prediction

error_and_exit $?

cnt=$(ls /data/gsgnn_lp_ml_dot/infer-emb/ | grep rel_emb.pt | wc -l)
if test $cnt -ne 0
then
    echo "Dot product inference does not output edge embedding"
    exit -1
fi

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train_embout /data/gsgnn_lp_ml_distmult/emb/ --infer_embout /data/gsgnn_lp_ml_distmult/infer-emb/ --link_prediction

error_and_exit $?

cnt=$(ls /data/gsgnn_lp_ml_distmult/infer-emb/ | grep rel_emb.pt | wc -l)
if test $cnt -ne 1
then
    echo "DistMult inference outputs edge embedding"
    exit -1
fi

cnt=$(ls /data/gsgnn_lp_ml_distmult/infer-emb/ | grep relation2id_map.json | wc -l)
if test $cnt -ne 1
then
    echo "DistMult inference outputs edge embedding"
    exit -1
fi

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, exclude_training_targets: true, save model, early stop"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --save-model-path /data/gsgnn_lp_ml_dot/ --topk-model-to-save 3 --save-model-per-iter 1000 --save-embed-path /data/gsgnn_lp_ml_dot/emb/ --enable-early-stop True --call-to-consider-early-stop 3 -e 30 --window-for-early-stop 2 --early-stop-strategy consecutive_increase" | tee exec.log

error_and_exit ${PIPESTATUS[0]}

# check early stop
cnt=$(cat exec.log | grep "Evaluation step" | wc -l)
if test $cnt -eq 30
then
	echo "Early stop should work, but it didn't"
	exit -1
fi

if test $cnt -le 4
then
	echo "Need at least 5 iters"
	exit -1
fi

rm -fr /data/gsgnn_lp_ml_dot/*

echo "**************dataset: Movielens, RGCN layer 1, BERT nodes: movie, user , inference: full-graph, negative_sampler: joint, decoder: DistMult, exclude_training_targets: true, save model"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp_text.yaml --fanout '10' --n-layers 1 --mini-batch-infer false  --use-node-embeddings true --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --save-model-path /data/gsgnn_lp_ml_distmult_text/ --topk-model-to-save 1 --save-model-per-iter 1000 --save-embed-path /data/gsgnn_lp_ml_distmult_text/emb/ --use-dot-product False --train-etype user,rating,movie movie,rating-rev,user" | tee train_log.txt

error_and_exit ${PIPESTATUS[0]}

best_epoch_distmult=$(grep "successfully save the model to" train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_distmult"

echo "**************dataset: Movielens text, do inference on saved model, decoder: DistMult"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/lp_infer --num_trainers $NUM_INFO_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --ip_config ip_list.txt --ssh_port 2222 "python3 lp_infer_gnn.py --cf ml_lp_text_infer.yaml --fanout '10' --n-layers 1 --mini-batch-infer false --use-node-embeddings true --num-gpus $NUM_INFO_TRAINERS --part-config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --save-embed-path /data/gsgnn_lp_ml_distmult_text/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_distmult_text/epoch-$best_epoch_distmult/ --use-dot-product False --no-validation True --train-etype user,rating,movie movie,rating-rev,user" | tee log2.txt

error_and_exit ${PIPESTATUS[0]}

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train_embout /data/gsgnn_lp_ml_distmult_text/emb/ --infer_embout /data/gsgnn_lp_ml_distmult_text/infer-emb/ --link_prediction

error_and_exit $?

cnt=$(ls /data/gsgnn_lp_ml_distmult_text/infer-emb/ | grep rel_emb.pt | wc -l)
if test $cnt -ne 1
then
    echo "DistMult inference outputs edge embedding"
    exit -1
fi

cnt=$(ls /data/gsgnn_lp_ml_distmult_text/infer-emb/ | grep relation2id_map.json | wc -l)
if test $cnt -ne 1
then
    echo "DistMult inference outputs edge embedding"
    exit -1
fi

rm -fr /data/gsgnn_lp_ml_distmult_text/*

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT, inference: full-graph, negative_sampler: joint, decoder: DistMult, exclude_training_targets: true, test_negative_sampler: joint"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --use-dot-product False --train-etype user,rating,movie movie,rating-rev,user"

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT, inference: full-graph, negative_sampler: joint, decoder: DistMult, exclude_training_targets: true, test_negative_sampler: uniform"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --use-dot-product False --train-etype user,rating,movie movie,rating-rev,user --test-negative-sampler uniform"

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT, inference: full-graph, negative_sampler: joint, exclude_training_targets: true, test_negative_sampler: uniform"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json  --test-negative-sampler uniform"

error_and_exit $?


echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, decoder: DistMult, exclude_training_targets: true, save model, train_etype: None"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp_none_train_etype.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --save-model-path /data/gsgnn_lp_ml_distmult_all_etype/ --topk-model-to-save 1 --save-model-per-iter 1000 --save-embed-path /data/gsgnn_lp_ml_distmult_all_etype/emb/ --use-dot-product False" | tee train_log.txt

error_and_exit ${PIPESTATUS[0]}

cnt=$(ls -l /data/gsgnn_lp_ml_distmult_all_etype/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi

best_epoch_distmult=$(grep "successfully save the model to" train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_distmult"

echo "**************dataset: Movielens, do inference on saved model, decoder: DistMult, eval_etype: None"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/lp_infer --num_trainers $NUM_INFO_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 lp_infer_gnn.py --cf ml_lp_none_train_etype_infer.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --num-gpus $NUM_INFO_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --save-embed-path /data/gsgnn_lp_ml_distmult_all_etype/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_distmult_all_etype/epoch-$best_epoch_distmult/ --use-dot-product False --no-validation True" | tee log2.txt

error_and_exit ${PIPESTATUS[0]}

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train_embout /data/gsgnn_lp_ml_distmult_all_etype/emb/ --infer_embout /data/gsgnn_lp_ml_distmult_all_etype/infer-emb/ --link_prediction

error_and_exit $?

rm -fr /data/gsgnn_lp_ml_distmult_all_etype/*

rm train_log.txt
echo "**************dataset: Movielens, Bert only, inference: full-graph, negative_sampler: joint, decoder: Dot, save model"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lm_lp.py --cf ml_lm_lp.yaml --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --save-model-path /data/gsgnn_lp_ml_lm_dot_all_etype/ --topk-model-to-save 1 --save-model-per-iter 1000 --save-embed-path /data/gsgnn_lp_ml_lm_dot_all_etype/emb/ --use-dot-product True" | tee train_log.txt

error_and_exit ${PIPESTATUS[0]}

cnt=$(ls -l /data/gsgnn_lp_ml_lm_dot_all_etype/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi

best_epoch_dot=$(grep "successfully save the model to" train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_dot"

echo "**************dataset: Movielens, Bert only, do inference on saved model, decoder: Dot, eval_etype: None"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/lp_infer --num_trainers $NUM_INFO_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --ip_config ip_list.txt --ssh_port 2222 "python3 lp_infer_lm.py --cf ml_lm_lp_infer.yaml --num-gpus $NUM_INFO_TRAINERS --part-config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --save-embed-path /data/gsgnn_lp_ml_lm_dot_all_etype/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_lm_dot_all_etype/epoch-$best_epoch_dot/ --use-dot-product True --no-validation True"

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train_embout /data/gsgnn_lp_ml_lm_dot_all_etype/emb/ --infer_embout /data/gsgnn_lp_ml_lm_dot_all_etype/infer-emb/ --link_prediction

error_and_exit $?

rm -fr /data/gsgnn_lp_ml_lm_dot_all_etype/*
rm train_log.txt


echo "**************dataset: Movielens, input encoder with Bert, inference: full-graph, negative_sampler: joint, decoder: Dot, save model"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lm_lp.py --cf ml_lm_lp.yaml --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --save-model-path /data/gsgnn_lp_ml_lmmlp_dot_all_etype/ --topk-model-to-save 1 --save-model-per-iter 1000 --save-embed-path /data/gsgnn_lp_ml_lmmlp_dot_all_etype/emb/ --use-dot-product True --model-encoder-type mlp" | tee train_log.txt

error_and_exit ${PIPESTATUS[0]}

cnt=$(ls -l /data/gsgnn_lp_ml_lmmlp_dot_all_etype/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi

best_epoch_dot=$(grep "successfully save the model to" train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_dot"

echo "**************dataset: Movielens, input encoder with Bert, do inference on saved model, decoder: Dot, eval_etype: None"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/lp_infer --num_trainers $NUM_INFO_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --ip_config ip_list.txt --ssh_port 2222 "python3 lp_infer_lm.py --cf ml_lm_lp_infer.yaml --num-gpus $NUM_INFO_TRAINERS --part-config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --save-embed-path /data/gsgnn_lp_ml_lmmlp_dot_all_etype/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_lmmlp_dot_all_etype/epoch-$best_epoch_dot/ --use-dot-product True --no-validation True --model-encoder-type mlp"

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train_embout /data/gsgnn_lp_ml_lmmlp_dot_all_etype/emb/ --infer_embout /data/gsgnn_lp_ml_lmmlp_dot_all_etype/infer-emb/ --link_prediction

error_and_exit $?

rm -fr /data/gsgnn_lp_ml_lmmlp_dot_all_etype/*

echo "**************dataset: Movielens, Bert only, inference: full-graph, negative_sampler: joint, decoder: Dot, save model, with lm-lr"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lm_lp.py --cf ml_lm_lp.yaml --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --save-model-path /data/gsgnn_lp_ml_lm_tune_lr_dot_all_etype/ --topk-model-to-save 1 --save-model-per-iter 1000 --save-embed-path /data/gsgnn_lp_ml_lm_tune_lr_dot_all_etype/emb/ --use-dot-product True --lm-tune-lr 0.0001" | tee train_log.txt

error_and_exit ${PIPESTATUS[0]}

cnt=$(ls -l /data/gsgnn_lp_ml_lm_tune_lr_dot_all_etype/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi

best_epoch_dot=$(grep "successfully save the model to" train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_dot"

echo "**************dataset: Movielens, Bert only, do inference on saved model, decoder: Dot, eval_etype: None, with lm-lr"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/lp_infer --num_trainers $NUM_INFO_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --ip_config ip_list.txt --ssh_port 2222 "python3 lp_infer_lm.py --cf ml_lm_lp_infer.yaml --num-gpus $NUM_INFO_TRAINERS --part-config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --save-embed-path /data/gsgnn_lp_ml_lm_tune_lr_dot_all_etype/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_lm_tune_lr_dot_all_etype/epoch-$best_epoch_dot/ --use-dot-product True --no-validation True"

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train_embout /data/gsgnn_lp_ml_lm_tune_lr_dot_all_etype/emb/ --infer_embout /data/gsgnn_lp_ml_lm_tune_lr_dot_all_etype/infer-emb/ --link_prediction

error_and_exit $?

rm -fr /data/gsgnn_lp_ml_lm_tune_lr_dot_all_etype/*

rm train_log.txt
echo "**************dataset: Movielens, input encoder with Bert, inference: full-graph, negative_sampler: joint, decoder: Dot, save model, with lm-lr"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lm_lp.py --cf ml_lm_lp.yaml --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --save-model-path /data/gsgnn_lp_ml_lmmlp_tune_lr_dot_all_etype/ --topk-model-to-save 1 --save-model-per-iter 1000 --save-embed-path /data/gsgnn_lp_ml_lmmlp_tune_lr_dot_all_etype/emb/ --use-dot-product True --model-encoder-type mlp --lm-tune-lr 0.0001" | tee train_log.txt

error_and_exit ${PIPESTATUS[0]}

cnt=$(ls -l /data/gsgnn_lp_ml_lmmlp_tune_lr_dot_all_etype/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi

best_epoch_dot=$(grep "successfully save the model to" train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_dot"

echo "**************dataset: Movielens, input encoder with Bert, do inference on saved model, decoder: Dot, eval_etype: None, with lm-lr"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/lp_infer --num_trainers $NUM_INFO_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --ip_config ip_list.txt --ssh_port 2222 "python3 lp_infer_lm.py --cf ml_lm_lp_infer.yaml --num-gpus $NUM_INFO_TRAINERS --part-config /data/movielen_100k_text_lp_train_val_1p_4t/movie-lens-100k-text.json --save-embed-path /data/gsgnn_lp_ml_lmmlp_tune_lr_dot_all_etype/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_lmmlp_tune_lr_dot_all_etype/epoch-$best_epoch_dot/ --use-dot-product True --no-validation True --model-encoder-type mlp"

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train_embout /data/gsgnn_lp_ml_lmmlp_tune_lr_dot_all_etype/emb/ --infer_embout /data/gsgnn_lp_ml_lmmlp_tune_lr_dot_all_etype/infer-emb/ --link_prediction

error_and_exit $?


rm -fr /data/gsgnn_lp_ml_lmmlp_tune_lr_dot_all_etype/*
