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

df /dev/shm -h

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, exclude_training_targets: true, save model"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false  --use-node-embeddings true --eval-batch-size 1024 --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie  --save-model-path /data/gsgnn_lp_ml_dot/ --topk-model-to-save 1 --save-model-frequency 1000 --save-embed-path /data/gsgnn_lp_ml_dot/emb/ --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True --eval-metric hit_at_1 hit_at_3 hit_at_10

error_and_exit $?

# check prints
cnt=$(grep "save_embed_path: /data/gsgnn_lp_ml_dot/emb/" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_embed_path"
    exit -1
fi

cnt=$(grep "save_model_path: /data/gsgnn_lp_ml_dot/" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_model_path"
    exit -1
fi

bst_cnt=$(grep "Best Test hit_at_1" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test hit@1"
    exit -1
fi

cnt=$(grep "| Test hit_at_1" /tmp/train_log.txt | wc -l)
if test $cnt -lt $bst_cnt
then
    echo "We use SageMaker task tracker, we should have Test hit@1"
    exit -1
fi

bst_cnt=$(grep "Best Validation hit_at_1" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation hit@1"
    exit -1
fi

cnt=$(grep "Validation hit_at_1" /tmp/train_log.txt | wc -l)
if test $cnt -lt $bst_cnt
then
    echo "We use SageMaker task tracker, we should have Validation hit@1"
    exit -1
fi

bst_cnt=$(grep "Best Test hit_at_3" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test hit@3"
    exit -1
fi

cnt=$(grep "| Test hit_at_3" /tmp/train_log.txt | wc -l)
if test $cnt -lt $bst_cnt
then
    echo "We use SageMaker task tracker, we should have Test hit@3"
    exit -1
fi

bst_cnt=$(grep "Best Validation hit_at_3" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation hit@3"
    exit -1
fi

cnt=$(grep "Validation hit_at_3" /tmp/train_log.txt | wc -l)
if test $cnt -lt $bst_cnt
then
    echo "We use SageMaker task tracker, we should have Validation hit@3"
    exit -1
fi

bst_cnt=$(grep "Best Test hit_at_10" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test hit@10"
    exit -1
fi

cnt=$(grep "| Test hit_at_10" /tmp/train_log.txt | wc -l)
if test $cnt -lt $bst_cnt
then
    echo "We use SageMaker task tracker, we should have Test hit@10"
    exit -1
fi

bst_cnt=$(grep "Best Validation hit_at_10" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation hit@10"
    exit -1
fi

cnt=$(grep "Validation hit_at_10" /tmp/train_log.txt | wc -l)
if test $cnt -lt $bst_cnt
then
    echo "We use SageMaker task tracker, we should have Validation hit@10"
    exit -1
fi

rm /tmp/train_log.txt

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, exclude_training_targets: true, save model"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false  --use-node-embeddings true --eval-batch-size 1024 --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie  --save-model-path /data/gsgnn_lp_ml_dot/ --topk-model-to-save 1 --save-model-frequency 1000 --save-embed-path /data/gsgnn_lp_ml_dot/emb/ --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True

error_and_exit $?

# check prints
cnt=$(grep "save_embed_path: /data/gsgnn_lp_ml_dot/emb/" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_embed_path"
    exit -1
fi

cnt=$(grep "save_model_path: /data/gsgnn_lp_ml_dot/" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_model_path"
    exit -1
fi

bst_cnt=$(grep "Best Test mrr" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test mrr"
    exit -1
fi

cnt=$(grep "| Test mrr" /tmp/train_log.txt | wc -l)
if test $cnt -lt $bst_cnt
then
    echo "We use SageMaker task tracker, we should have Test mrr"
    exit -1
fi

bst_cnt=$(grep "Best Validation mrr" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation mrr"
    exit -1
fi

cnt=$(grep "Validation mrr" /tmp/train_log.txt | wc -l)
if test $cnt -lt $bst_cnt
then
    echo "We use SageMaker task tracker, we should have Validation mrr"
    exit -1
fi

cnt=$(grep "Best Iteration" /tmp/train_log.txt | wc -l)
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

cnt=$(ls -l /data/gsgnn_lp_ml_dot/emb/user | grep parquet | wc -l)
if test $cnt != $NUM_TRAINERS
then
    echo "The number of remapped user embeddings $cnt is not equal to the number of trainers $NUM_TRAINERS"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_lp_ml_dot/emb/movie | grep parquet | wc -l)
if test $cnt != $NUM_TRAINERS
then
    echo "The number of remapped movie embeddings $cnt is not equal to the number of trainers $NUM_TRAINERS"
    exit -1
fi

best_epoch_dot=$(grep "successfully save the model to" /tmp/train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_dot"

rm /tmp/train_log.txt

cnt=$(ls /data/gsgnn_lp_ml_dot/epoch-$best_epoch_dot/user/ | wc -l)
if test $cnt != 4
then
    echo "The number of sparse emb files $cnt is not equal to the number of gpus 4"
    exit -1
fi

cnt=$(ls /data/gsgnn_lp_ml_dot/epoch-$best_epoch_dot/movie/ | wc -l)
if test $cnt != 4
then
    echo "The number of sparse emb files $cnt is not equal to the number of gpus 4"
    exit -1
fi

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, decoder: DistMult, exclude_training_targets: true, save model"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false  --use-node-embeddings true  --eval-batch-size 1024 --save-model-path /data/gsgnn_lp_ml_distmult/ --topk-model-to-save 1 --save-model-frequency 1000 --save-embed-path /data/gsgnn_lp_ml_distmult/emb/ --lp-decoder-type distmult --train-etype user,rating,movie movie,rating-rev,user --logging-file /tmp/train_log.txt --preserve-input True

error_and_exit $?

cnt=$(ls -l /data/gsgnn_lp_ml_distmult/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi

best_epoch_distmult=$(grep "successfully save the model to" /tmp/train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_distmult"

rm /tmp/train_log.txt

cnt=$(ls /data/gsgnn_lp_ml_distmult/epoch-$best_epoch_distmult/user/ | wc -l)
if test $cnt != 4
then
    echo "The number of sparse emb files $cnt is not equal to the number of gpus 4"
    exit -1
fi

cnt=$(ls /data/gsgnn_lp_ml_distmult/epoch-$best_epoch_distmult/movie/ | wc -l)
if test $cnt != 4
then
    echo "The number of sparse emb files $cnt is not equal to the number of gpus 4"
    exit -1
fi

echo "**************dataset: Movielens, do inference on saved model, decoder: dot"
python3 -m graphstorm.run.gs_link_prediction --inference --workspace $GS_HOME/inference_scripts/lp_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp_infer.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --use-node-embeddings true --eval-batch-size 1024 --save-embed-path /data/gsgnn_lp_ml_dot/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_dot/epoch-$best_epoch_dot/ --logging-file /tmp/log.txt --preserve-input True

error_and_exit $?

cnt=$(grep "| Test mrr" /tmp/log.txt | wc -l)
if test $cnt -ne 1
then
    echo "We do test, should have mrr"
    exit -1
fi

bst_cnt=$(grep "Best Test mrr" /tmp/log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test mrr"
    exit -1
fi

bst_cnt=$(grep "Best Validation mrr" /tmp/log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation mrr"
    exit -1
fi

cnt=$(grep "Validation mrr" /tmp/log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Validation mrr"
    exit -1
fi

cnt=$(grep "Best Iteration" /tmp/log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Iteration"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_lp_ml_dot/infer-emb/user | grep parquet | wc -l)
if test $cnt != $NUM_INFO_TRAINERS
then
    echo "The number of remapped user embeddings $cnt is not equal to the number of inferers $NUM_INFO_TRAINERS"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_lp_ml_dot/infer-emb/movie | grep parquet | wc -l)
if test $cnt != $NUM_INFO_TRAINERS
then
    echo "The number of remapped movie embeddings $cnt is not equal to the number of inferers $NUM_INFO_TRAINERS"
    exit -1
fi

rm /tmp/log.txt

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_lp_ml_dot/emb/ --infer-embout /data/gsgnn_lp_ml_dot/infer-emb/ --link-prediction

error_and_exit $?

cnt=$(ls /data/gsgnn_lp_ml_dot/infer-emb/ | grep rel_emb.pt | wc -l)
if test $cnt -ne 0
then
    echo "Dot product inference does not output edge embedding"
    exit -1
fi
rm -fr /data/gsgnn_lp_ml_dot/infer-emb/

echo "**************dataset: Movielens, do inference on saved model, decoder: dot, remap without shared file system"
python3 -m graphstorm.run.gs_link_prediction --inference --workspace $GS_HOME/inference_scripts/lp_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp_infer.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --use-node-embeddings true --eval-batch-size 1024 --save-embed-path /data/gsgnn_lp_ml_dot/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_dot/epoch-$best_epoch_dot/ --logging-file /tmp/log.txt --preserve-input True --with-shared-fs False

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_lp_ml_dot/emb/ --infer-embout /data/gsgnn_lp_ml_dot/infer-emb/ --link-prediction

error_and_exit $?

rm -fr /data/gsgnn_lp_ml_dot/infer-emb/
rm /tmp/log.txt

echo "**************dataset: Movielens, use gen_embeddings to generate embeddings on link prediction"
python3 -m graphstorm.run.gs_gen_node_embedding --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --eval-batch-size 1024 --use-node-embeddings true --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --restore-model-path /data/gsgnn_lp_ml_dot/epoch-$best_epoch_dot/ --save-embed-path /data/gsgnn_lp_ml_dot/save-emb/ --logging-file /tmp/train_log.txt --logging-level debug

error_and_exit $?

cnt=$(ls /data/gsgnn_lp_ml_dot/save-emb/movie | grep .pt | wc -l)
if test $cnt -ne 0
then
    echo "/data/gsgnn_lp_ml_dot/save-emb/movie/embed-0000x.pt should be removed."
    exit -1
fi

cnt=$(ls /data/gsgnn_lp_ml_dot/save-emb/movie | grep .parquet | wc -l)
if test $cnt -ne $NUM_TRAINERS
then
    echo "$NUM_TRAINERS /data/gsgnn_lp_ml_dot/save-emb/movie/embed-0000x_0000x.parquet files must exist."
    exit -1
fi

cnt=$(ls /data/gsgnn_lp_ml_dot/save-emb/user | grep .pt | wc -l)
if test $cnt -ne 0
then
    echo "/data/gsgnn_lp_ml_dot/save-emb/user/embed-0000x.pt should be removed."
    exit -1
fi

cnt=$(ls /data/gsgnn_lp_ml_dot/save-emb/user | grep .parquet | wc -l)
if test $cnt -ne $NUM_TRAINERS
then
    echo "$NUM_TRAINERS /data/gsgnn_lp_ml_dot/save-emb/user/embed-0000x_0000x.parquet files must exist."
    exit -1
fi

echo "**************dataset: Movielens, do mini-batch inference on saved model, decoder: dot"
python3 -m graphstorm.run.gs_link_prediction --inference --workspace $GS_HOME/inference_scripts/lp_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp_infer.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --use-node-embeddings true --eval-batch-size 1024 --save-embed-path /data/gsgnn_lp_ml_dot/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_dot/epoch-$best_epoch_dot/ --use-mini-batch-infer true --logging-file /tmp/log.txt --preserve-input True

error_and_exit $?

cnt=$(grep "| Test mrr" /tmp/log.txt | wc -l)
if test $cnt -ne 1
then
    echo "We do test, should have mrr"
    exit -1
fi

bst_cnt=$(grep "Best Test mrr" /tmp/log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test mrr"
    exit -1
fi

bst_cnt=$(grep "Best Validation mrr" /tmp/log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation mrr"
    exit -1
fi

cnt=$(grep "Validation mrr" /tmp/log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Validation mrr"
    exit -1
fi

cnt=$(grep "Best Iteration" /tmp/log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Iteration"
    exit -1
fi

rm /tmp/log.txt

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_lp_ml_dot/emb/ --infer-embout /data/gsgnn_lp_ml_dot/infer-emb/ --link-prediction

error_and_exit $?
rm -fr /data/gsgnn_lp_ml_dot/infer-emb/

echo "**************dataset: Movielens, do inference on saved model, decoder: DistMult"
python3 -m graphstorm.run.gs_link_prediction --inference --workspace $GS_HOME/inference_scripts/lp_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp_infer.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --use-node-embeddings true --eval-batch-size 1024 --save-embed-path /data/gsgnn_lp_ml_distmult/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_distmult/epoch-$best_epoch_distmult/ --lp-decoder-type distmult --no-validation False --train-etype user,rating,movie movie,rating-rev,user --preserve-input True

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_lp_ml_distmult/emb/ --infer-embout /data/gsgnn_lp_ml_distmult/infer-emb/ --link-prediction

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
python3 -m graphstorm.run.launch --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 $GS_HOME/python/graphstorm/run/gsgnn_lp/gsgnn_lp.py --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false  --use-node-embeddings true --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --save-model-path /data/gsgnn_lp_ml_dot/ --topk-model-to-save 3 --save-model-frequency 1000 --save-embed-path /data/gsgnn_lp_ml_dot/emb/ --use-early-stop True --early-stop-burnin-rounds 3 -e 30 --early-stop-rounds 2 --early-stop-strategy consecutive_increase --logging-file /tmp/exec.log

error_and_exit $?

# check early stop
cnt=$(cat /tmp/exec.log | grep "Evaluation step" | wc -l)
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

rm /tmp/exec.log

rm -fr /data/gsgnn_lp_ml_dot/*

echo "**************dataset: Movielens, RGCN layer 1, BERT nodes: movie, user , inference: full-graph, negative_sampler: joint, decoder: Dot Product, exclude_training_targets: true, save model"
python3 -m graphstorm.run.launch --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lm_encoder_lp_train_val_1p_4t/movie-lens-100k-text.json --ip-config ip_list.txt --ssh-port 2222 --preserve-input True $GS_HOME/python/graphstorm/run/gsgnn_lp/gsgnn_lp.py --cf ml_lp_text.yaml --fanout '4' --num-layers 1 --use-mini-batch-infer false  --use-node-embeddings true --save-model-path /data/gsgnn_lp_ml_dotprod_text/ --topk-model-to-save 1 --save-model-frequency 1000 --save-embed-path /data/gsgnn_lp_ml_dotprod_text/emb/ --lp-decoder-type dot_product --train-etype user,rating,movie --logging-file /tmp/train_log.txt

error_and_exit $?

best_epoch_dotprod=$(grep "successfully save the model to" /tmp/train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_dotprod"

rm /tmp/train_log.txt

echo "**************dataset: Movielens text, do inference on saved model, decoder: Dot Product"
python3 -m graphstorm.run.launch --workspace $GS_HOME/inference_scripts/lp_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lm_encoder_lp_train_val_1p_4t/movie-lens-100k-text.json --ip-config ip_list.txt --ssh-port 2222 --preserve-input True $GS_HOME/python/graphstorm/run/gsgnn_lp/lp_infer_gnn.py --cf ml_lp_text_infer.yaml --fanout '4' --num-layers 1 --use-mini-batch-infer false --use-node-embeddings true   --save-embed-path /data/gsgnn_lp_ml_dotprod_text/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_dotprod_text/epoch-$best_epoch_dotprod/ --lp-decoder-type dot_product --no-validation False --train-etype user,rating,movie

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_lp_ml_dotprod_text/emb/ --infer-embout /data/gsgnn_lp_ml_dotprod_text/infer-emb/ --link-prediction

error_and_exit $?

rm -fr /data/gsgnn_lp_ml_dotprod_text/*

echo "**************dataset: Movielens, RGCN layer 1, BERT/ALBERT nodes: movie, user (different hidden dims), inference: mini-batch, negative_sampler: joint, decoder: Dot Product, exclude_training_targets: true, save model"
python3 -m graphstorm.run.launch --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lm_encoder_lp_train_val_1p_4t/movie-lens-100k-text.json --ip-config ip_list.txt --ssh-port 2222 $GS_HOME/python/graphstorm/run/gsgnn_lp/gsgnn_lp.py --cf ml_lp_text_multiple_lm_models.yaml --fanout '4' --num-layers 1 --lp-decoder-type dot_product --train-etype user,rating,movie

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, inference: full-graph, negative_sampler: localuniform, exclude_training_targets: true, test_negative_sampler: uniform"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '5' --num-layers 1 --use-mini-batch-infer false --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --train-negative-sampler localuniform --eval-negative-sampler uniform

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, inference: full-graph, negative_sampler: localjoint, exclude_training_targets: true, test_negative_sampler: uniform"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '5' --num-layers 1 --use-mini-batch-infer false --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --train-negative-sampler localjoint --eval-negative-sampler uniform

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, inference: full-graph, negative_sampler: fast_uniform, exclude_training_targets: true, test_negative_sampler: uniform"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '5' --num-layers 1 --use-mini-batch-infer false --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --train-negative-sampler fast_uniform --eval-negative-sampler uniform

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, inference: full-graph, negative_sampler: fast_joint, exclude_training_targets: true, test_negative_sampler: uniform"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '5' --num-layers 1 --use-mini-batch-infer false --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --train-negative-sampler fast_joint --eval-negative-sampler uniform

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, inference: full-graph, negative_sampler: fast_localuniform, exclude_training_targets: true, test_negative_sampler: uniform"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '5' --num-layers 1 --use-mini-batch-infer false --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --train-negative-sampler fast_localuniform --eval-negative-sampler uniform

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, inference: full-graph, negative_sampler: fast_localjoint, exclude_training_targets: true, test_negative_sampler: uniform"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '5' --num-layers 1 --use-mini-batch-infer false --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --train-negative-sampler fast_localjoint --eval-negative-sampler uniform

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, decoder: DistMult, exclude_training_targets: true, save model, train_etype: None"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp_none_train_etype.yaml --fanout '5' --num-layers 1 --use-mini-batch-infer false  --use-node-embeddings true --save-model-path /data/gsgnn_lp_ml_distmult_all_etype/ --topk-model-to-save 1 --save-model-frequency 1000 --save-embed-path /data/gsgnn_lp_ml_distmult_all_etype/emb/ --lp-decoder-type distmult --logging-file /tmp/train_log.txt --preserve-input True

error_and_exit $?

cnt=$(ls -l /data/gsgnn_lp_ml_distmult_all_etype/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi

best_epoch_distmult=$(grep "successfully save the model to" /tmp/train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_distmult"

rm /tmp/train_log.txt

echo "**************dataset: Movielens, do inference on saved model, decoder: DistMult, eval_etype: None"
python3 -m graphstorm.run.gs_link_prediction --inference --workspace $GS_HOME/inference_scripts/lp_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp_none_train_etype_infer.yaml --fanout '5' --num-layers 1 --use-mini-batch-infer false  --use-node-embeddings true  --save-embed-path /data/gsgnn_lp_ml_distmult_all_etype/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_distmult_all_etype/epoch-$best_epoch_distmult/ --lp-decoder-type distmult --no-validation False --preserve-input True

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_lp_ml_distmult_all_etype/emb/ --infer-embout /data/gsgnn_lp_ml_distmult_all_etype/infer-emb/ --link-prediction

error_and_exit $?

rm -fr /data/gsgnn_lp_ml_distmult_all_etype/*

echo "**************dataset: Movielens, Bert only, inference: full-graph, negative_sampler: joint, decoder: Dot, save model"
python3 -m graphstorm.run.gs_link_prediction --lm-encoder-only --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lm_encoder_lp_train_val_1p_4t/movie-lens-100k-text.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lm_lp.yaml --save-model-path /data/gsgnn_lp_ml_lm_dot_all_etype/ --topk-model-to-save 1 --save-model-frequency 1000 --save-embed-path /data/gsgnn_lp_ml_lm_dot_all_etype/emb/ --lp-decoder-type dot_product --logging-file /tmp/train_log.txt --preserve-input True --backend nccl

error_and_exit $?

cnt=$(ls -l /data/gsgnn_lp_ml_lm_dot_all_etype/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi

best_epoch_dot=$(grep "successfully save the model to" /tmp/train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_dot"

rm /tmp/train_log.txt

echo "**************dataset: Movielens, Bert only, do inference on saved model, decoder: Dot, eval_etype: None"
python3 -m graphstorm.run.gs_link_prediction --lm-encoder-only --inference --workspace $GS_HOME/inference_scripts/lp_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lm_encoder_lp_train_val_1p_4t/movie-lens-100k-text.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lm_lp_infer.yaml   --save-embed-path /data/gsgnn_lp_ml_lm_dot_all_etype/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_lm_dot_all_etype/epoch-$best_epoch_dot/ --lp-decoder-type dot_product --no-validation True --preserve-input True  --backend nccl

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_lp_ml_lm_dot_all_etype/emb/ --infer-embout /data/gsgnn_lp_ml_lm_dot_all_etype/infer-emb/ --link-prediction

error_and_exit $?

rm -fr /data/gsgnn_lp_ml_lm_dot_all_etype/*


echo "**************dataset: Movielens, input encoder with Bert, inference: full-graph, negative_sampler: joint, decoder: Dot, save model"
python3 -m graphstorm.run.launch --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lm_encoder_lp_train_val_1p_4t/movie-lens-100k-text.json --ip-config ip_list.txt --ssh-port 2222 --preserve-input True $GS_HOME/python/graphstorm/run/gsgnn_lp/gsgnn_lm_lp.py --cf ml_lm_lp.yaml  --save-model-path /data/gsgnn_lp_ml_lmmlp_dot_all_etype/ --topk-model-to-save 1 --save-model-frequency 1000 --save-embed-path /data/gsgnn_lp_ml_lmmlp_dot_all_etype/emb/ --lp-decoder-type dot_product --model-encoder-type mlp --logging-file /tmp/train_log.txt

error_and_exit $?

cnt=$(ls -l /data/gsgnn_lp_ml_lmmlp_dot_all_etype/ | grep epoch | wc -l)
if test $cnt != 1
then
    echo "The number of save models $cnt is not equal to the specified topk 1"
    exit -1
fi

best_epoch_dot=$(grep "successfully save the model to" /tmp/train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_dot"

rm /tmp/train_log.txt

echo "**************dataset: Movielens, input encoder with Bert, do inference on saved model, decoder: Dot, eval_etype: None"
python3 -m graphstorm.run.launch --workspace $GS_HOME/inference_scripts/lp_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lm_encoder_lp_train_val_1p_4t/movie-lens-100k-text.json --ip-config ip_list.txt --ssh-port 2222 --preserve-input True $GS_HOME/python/graphstorm/run/gsgnn_lp/lp_infer_lm.py --cf ml_lm_lp_infer.yaml   --save-embed-path /data/gsgnn_lp_ml_lmmlp_dot_all_etype/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_lmmlp_dot_all_etype/epoch-$best_epoch_dot/ --lp-decoder-type dot_product --no-validation True --model-encoder-type mlp

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_lp_ml_lmmlp_dot_all_etype/emb/ --infer-embout /data/gsgnn_lp_ml_lmmlp_dot_all_etype/infer-emb/ --link-prediction

error_and_exit $?

rm -fr /data/gsgnn_lp_ml_lmmlp_dot_all_etype/*

# Test sample with edge weight
echo "**************dataset: Movielens, only one training edge with edge weight for loss***********"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false  --use-node-embeddings true --eval-batch-size 1024 --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie  --save-model-path /data/gsgnn_lp_ml_dot/ --topk-model-to-save 1 --save-model-frequency 1000 --save-embed-path /data/gsgnn_lp_ml_dot/emb/ --lp-edge-weight-for-loss rate

error_and_exit $?

echo "**************dataset: Movielens, two training edges but only one with edge weight for loss***********"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false  --use-node-embeddings true --eval-batch-size 1024  --save-model-path /data/gsgnn_lp_ml_dot/ --topk-model-to-save 1 --save-model-frequency 1000 --save-embed-path /data/gsgnn_lp_ml_dot/emb/ --train-etype user,rating,movie movie,rating-rev,user --lp-edge-weight-for-loss user,rating,movie:rate

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed feature, inference: full-graph, negative_sampler: joint, exclude_training_targets: true, save model, Backend nccl"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10' --num-layers 1 --use-mini-batch-infer false --eval-batch-size 128 --reverse-edge-types-map user,rating,rating-rev,movie --node-feat-name movie:title user:feat --backend nccl

error_and_exit $?

echo "**************dataset: Movielens, two training edges with per etype evaluation result***********"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_2etype_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10' --num-layers 1 --use-mini-batch-infer false --eval-batch-size 1024 --exclude-training-targets false --train-etype user,rating,movie user,rating2,movie --eval-etype user,rating,movie user,rating2,movie --report-eval-per-type True --node-feat-name movie:title user:feat --logging-file /tmp/train_log.txt

error_and_exit $?

cnt=$(grep "Test mrr: {('user', 'rating', 'movie'):" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "Should have Test mrr: {('user', 'rating', 'movie'):"
    exit -1
fi
rm /tmp/train_log.txt

echo "**************dataset: Movielens, two training edges with per etype evaluation result***********"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_2etype_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10' --num-layers 1 --use-mini-batch-infer false --eval-batch-size 1024 --exclude-training-targets false --train-etype user,rating,movie user,rating2,movie --eval-etype user,rating,movie user,rating2,movie --model-select-etype user,rating2,movie --report-eval-per-type True --node-feat-name movie:title user:feat --logging-file /tmp/train_log.txt

error_and_exit $?

cnt=$(grep "Test mrr: {('user', 'rating', 'movie'):" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "Should have Test mrr: {('user', 'rating', 'movie')"
    exit -1
fi

cnt=$(grep "('user', 'rating2', 'movie'):" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "Should have ('user', 'rating2', 'movie') in validation and test"
    exit -1
fi
rm /tmp/train_log.txt

echo "**************dataset: Movielens, input encoder with Bert, inference: full-graph, negative_sampler: joint, decoder: Dot, save model"
python3 -m graphstorm.run.launch --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lm_encoder_lp_train_val_1p_4t/movie-lens-100k-text.json --ip-config ip_list.txt --ssh-port 2222 $GS_HOME/python/graphstorm/run/gsgnn_lp/gsgnn_lm_lp.py --cf ml_lm_lp.yaml  --lp-decoder-type dot_product --model-encoder-type mlp --report-eval-per-type True --num-epochs 1

error_and_exit $?

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, negative_sampler: joint, exclude_training_targets: true, save model, enough hard neg"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_hard_neg_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false  --eval-batch-size 1024 --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie  --save-model-path /data/gsgnn_lp_ml_hard_dot/  --save-model-frequency 1000 --train-etypes-negative-dstnode hard_0 --num-train-hard-negatives 4 --num-negative-edges 10 --target-etype user,rating,movie

error_and_exit $?

echo "**************dataset: Movielens, do inference on saved model, decoder: dot with fixed negative"
python3 -m graphstorm.run.gs_link_prediction --inference --workspace $GS_HOME/inference_scripts/lp_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_hard_neg_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp_infer.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --eval-batch-size 1024 --restore-model-path /data/gsgnn_lp_ml_hard_dot/epoch-2/ --eval-etypes-negative-dstnode fixed_eval --eval-etype user,rating,movie

error_and_exit $?

rm -fr /data/gsgnn_lp_ml_hard_dot/*

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, negative_sampler: joint, exclude_training_targets: true, save model, hard neg + random neg"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_hard_neg_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false  --eval-batch-size 1024 --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie  --save-model-path /data/gsgnn_lp_ml_hard_dot/  --save-model-frequency 1000 --train-etypes-negative-dstnode user,rating,movie:hard_1 --num-train-hard-negatives 5 --num-negative-edges 10 --target-etype user,rating,movie

error_and_exit $?

echo "**************dataset: Movielens, do inference on saved model, decoder: dot with fixed negative"
python3 -m graphstorm.run.gs_link_prediction --inference --workspace $GS_HOME/inference_scripts/lp_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_hard_neg_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp_infer.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --eval-batch-size 1024 --restore-model-path /data/gsgnn_lp_ml_hard_dot/epoch-2/ --eval-etypes-negative-dstnode user,rating,movie:fixed_eval --eval-etype user,rating,movie

error_and_exit $?

rm -fr /data/gsgnn_lp_ml_hard_dot/*

# wholegraph sparse embedding
echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, exclude_training_targets: true, save model, wholegraph learnable emb"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false  --use-node-embeddings true --eval-batch-size 1024 --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie  --save-model-path /data/gsgnn_lp_ml_wg_dot/ --topk-model-to-save 1 --save-model-frequency 1000 --save-embed-path /data/gsgnn_lp_ml_wg_dot/emb/ --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True --use-wholegraph-embed True  --backend nccl

error_and_exit $?

# check prints
cnt=$(grep "save_embed_path: /data/gsgnn_lp_ml_wg_dot/emb/" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_embed_path"
    exit -1
fi

cnt=$(grep "save_model_path: /data/gsgnn_lp_ml_wg_dot/" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_model_path"
    exit -1
fi

bst_cnt=$(grep "Best Test mrr" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test mrr"
    exit -1
fi

best_epoch_dot=$(grep "successfully save the model to" /tmp/train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_dot"

cnt=$(ls /data/gsgnn_lp_ml_wg_dot/epoch-$best_epoch_dot/user/ | wc -l)
if test $cnt != 4
then
    echo "The number of sparse emb files $cnt is not equal to the number of gpus 4"
    exit -1
fi

cnt=$(ls /data/gsgnn_lp_ml_wg_dot/epoch-$best_epoch_dot/movie/ | wc -l)
if test $cnt != 4
then
    echo "The number of sparse emb files $cnt is not equal to the number of gpus 4"
    exit -1
fi

echo "**************dataset: Movielens, do inference on saved model, decoder: dot, wholegraph learnable emb"
python3 -m graphstorm.run.gs_link_prediction --inference --workspace $GS_HOME/inference_scripts/lp_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp_infer.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --use-node-embeddings true --eval-batch-size 1024 --save-embed-path /data/gsgnn_lp_ml_wg_dot/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_wg_dot/epoch-$best_epoch_dot/ --logging-file /tmp/log.txt --preserve-input True --use-wholegraph-embed True  --backend nccl

error_and_exit $?

bst_cnt=$(grep "Best Test mrr" /tmp/log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test mrr"
    exit -1
fi

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_lp_ml_wg_dot/emb/ --infer-embout /data/gsgnn_lp_ml_wg_dot/infer-emb/ --link-prediction

error_and_exit $?

rm -fr /data/gsgnn_lp_ml_wg_dot/
rm /tmp/train_log.txt


echo "=================== test save model and do evaluation behaviors ==================="

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, inference: full-graph, negative_sampler: localuniform, exclude_training_targets: true, test_negative_sampler: uniform, no-topk save model, no eval frequency"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '5' --num-layers 1 --use-mini-batch-infer false --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --batch-size 128  --save-model-frequency 10 --save-model-path /data/gsgnn_lp_ml_ns_lu/  --num-epochs 1 --logging-file /tmp/train_log.txt

error_and_exit $?

save_model_cnts=$(grep "successfully save the model to" /tmp/train_log.txt | wc -l)
do_eval_cnts=$(grep "Best Validation" /tmp/train_log.txt | wc -l)

if [ $save_model_cnts != 3 ] || [ $do_eval_cnts != 3 ]
then
    echo "The number of save models is not equal to the number of do evaluation and not equal to 3, but got $save_model_cnts and $do_eval_cnts."
    exit -1
fi

rm /tmp/train_log.txt

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, inference: full-graph, negative_sampler: localuniform, exclude_training_targets: true, test_negative_sampler: uniform, no-topk save model, eval less frequently but divisible by save model frequency"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '5' --num-layers 1 --use-mini-batch-infer false --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --batch-size 128 --save-model-frequency 10 --eval-frequency 20 --save-model-path /data/gsgnn_lp_ml_ns_lu/  --num-epochs 1 --logging-file /tmp/train_log.txt

error_and_exit $?

save_model_cnts=$(grep "successfully save the model to" /tmp/train_log.txt | wc -l)
do_eval_cnts=$(grep "Best Validation" /tmp/train_log.txt | wc -l)

if [ $save_model_cnts != 3 ] || [ $do_eval_cnts != 3 ]
then
    echo "The number of save models is not equal to the number of do evaluation and not equal to 3, but got $save_model_cnts and $do_eval_cnts."
    exit -1
fi

rm /tmp/train_log.txt

echo "**************dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, inference: full-graph, negative_sampler: localuniform, exclude_training_targets: true, test_negative_sampler: uniform, no-topk save model, eval more frequently"
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '5' --num-layers 1 --use-mini-batch-infer false --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --batch-size 128 --save-model-frequency 10 --eval-frequency 5 --save-model-path /data/gsgnn_lp_ml_ns_lu/  --num-epochs 1 --logging-file /tmp/train_log.txt

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

rm -fr /tmp/*
