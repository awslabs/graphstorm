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
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --save-model-path /data/gsgnn_lp_ml_dot/ --topk-model-to-save 3 --save-model-per-iter 1000 --save-embeds-path /data/gsgnn_lp_ml_dot/emb/" | tee train_log.txt

error_and_exit $?

# check prints
cnt=$(grep "save_embeds_path: /data/gsgnn_lp_ml_dot/emb/" train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_embeds_path"
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
if test $cnt != 3
then
    echo "The number of save models $cnt is not equal to the specified topk 3"
    exit -1
fi

echo "**************dataset: Movielens, RGCN layer 2, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, negative_sampler: joint, decoder: DistMult, exclude_training_targets: true, save model"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --save-model-path /data/gsgnn_lp_ml_distmult/ --topk-model-to-save 3 --save-model-per-iter 1000 --save-embeds-path /data/gsgnn_lp_ml_distmult/emb/ --use-dot-product False --train-etype user,rating,movie movie,rating-rev,user"

error_and_exit $?

echo "**************dataset: Movielens, do inference on saved model, decoder: dot"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/lp_infer --num_trainers $NUM_INFO_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 lp_infer_gnn.py --cf ml_lp_infer.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --num-gpus $NUM_INFO_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --save-embeds-path /data/gsgnn_lp_ml_dot/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_dot/epoch-2/" | tee log.txt

error_and_exit $?

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
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/lp_infer --num_trainers $NUM_INFO_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 lp_infer_gnn.py --cf ml_lp_infer.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --num-gpus $NUM_INFO_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --save-embeds-path /data/gsgnn_lp_ml_distmult/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_distmult/epoch-2/ --use-dot-product False --no-validation True --train-etype user,rating,movie movie,rating-rev,user" | tee log2.txt

error_and_exit $?

cd $GS_HOME/tests/end2end-tests/graphstorm-lp/
python3 $GS_HOME/tests/end2end-tests/check_infer.py --train_embout /data/gsgnn_lp_ml_dot/emb/epoch-2/ --infer_embout /data/gsgnn_lp_ml_dot/infer-emb/

error_and_exit $?

cnt=$(ls /data/gsgnn_lp_ml_dot/infer-emb/ | grep rel_emb.pt | wc -l)
if test $cnt -ne 0
then
    echo "Dot product inference does not output edge embedding"
    exit -1
fi

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train_embout /data/gsgnn_lp_ml_distmult/emb/epoch-2/ --infer_embout /data/gsgnn_lp_ml_distmult/infer-emb/

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
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --save-model-path /data/gsgnn_lp_ml_dot/ --topk-model-to-save 3 --save-model-per-iter 1000 --save-embeds-path /data/gsgnn_lp_ml_dot/emb/ --enable-early-stop True --call-to-consider-early-stop 3 -e 30 --window-for-early-stop 2 --early-stop-strategy consecutive_increase" | tee exec.log

error_and_exit $?

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
