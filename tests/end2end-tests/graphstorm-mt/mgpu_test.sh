#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=4
NUM_INFO_TRAINERS=2
NUM_INFERs=2
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_mt
echo "127.0.0.1" > ip_list.txt
cd $GS_HOME/inference_scripts/mt_infer
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


echo "**************[Multi-task] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, save model"
python3 -m graphstorm.run.gs_multi_task_learning --workspace $GS_HOME/training_scripts/gsgnn_mt  --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_ec_er_lp.yaml --save-model-path /data/gsgnn_mt/ --save-model-frequency 1000 --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True

error_and_exit $?

# check prints

bst_cnt=$(grep "Best Test node_classification" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test node_classification"
    exit -1
fi

cnt=$(grep "Test node_classification" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Test node_classification"
    exit -1
fi

bst_cnt=$(grep "Best Validation node_classification" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation accuracy node_classification"
    exit -1
fi

cnt=$(grep "Validation node_classification" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Validation node_classification"
    exit -1
fi

bst_cnt=$(grep "Best Test edge_classification" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test edge_classification"
    exit -1
fi

cnt=$(grep "Test edge_classification" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Test edge_classification"
    exit -1
fi

bst_cnt=$(grep "Best Validation edge_classification" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation edge_classification"
    exit -1
fi

cnt=$(grep "Validation edge_classification" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Validation edge_classification"
    exit -1
fi

bst_cnt=$(grep "Best Test edge_regression" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test edge_regression"
    exit -1
fi

cnt=$(grep "Test edge_regression" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Test edge_regression"
    exit -1
fi

bst_cnt=$(grep "Best Validation edge_regression" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation edge_regression"
    exit -1
fi

cnt=$(grep "Validation edge_regression" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Validation edge_regression"
    exit -1
fi

bst_cnt=$(grep "Best Test link_prediction" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test link_prediction"
    exit -1
fi

cnt=$(grep "Test link_prediction" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Test link_prediction"
    exit -1
fi

bst_cnt=$(grep "Best Validation link_prediction" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation link_prediction"
    exit -1
fi

cnt=$(grep "Validation link_prediction" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Validation link_prediction"
    exit -1
fi

bst_cnt=$(grep "Best Test reconstruct_node_feat" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test reconstruct_node_feat"
    exit -1
fi

cnt=$(grep "Test reconstruct_node_feat" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Test reconstruct_node_feat"
    exit -1
fi

bst_cnt=$(grep "Best Validation reconstruct_node_feat" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation reconstruct_node_feat"
    exit -1
fi

cnt=$(grep "Validation reconstruct_node_feat" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Validation reconstruct_node_feat"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_mt/ | grep epoch | wc -l)
if test $cnt != 3
then
    echo "The number of save models $cnt is not equal to the specified topk 3"
    exit -1
fi

rm -fr /data/gsgnn_mt/
rm /tmp/train_log.txt

echo "**************[Multi-task] dataset: Movielens, RGAT layer 2, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, save model"
python3 -m graphstorm.run.gs_multi_task_learning --workspace $GS_HOME/training_scripts/gsgnn_mt  --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_ec_er_lp.yaml --save-model-path /data/gsgnn_mt/ --save-model-frequency 1000 --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True --num-layers 2 --fanout "4,4" --model-encoder-type rgat

error_and_exit $?

# check prints

bst_cnt=$(grep "Best Test node_classification" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test node_classification"
    exit -1
fi

cnt=$(grep "Test node_classification" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Test node_classification"
    exit -1
fi

bst_cnt=$(grep "Best Validation node_classification" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation accuracy node_classification"
    exit -1
fi

cnt=$(grep "Validation node_classification" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Validation node_classification"
    exit -1
fi

bst_cnt=$(grep "Best Test edge_classification" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test edge_classification"
    exit -1
fi

cnt=$(grep "Test edge_classification" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Test edge_classification"
    exit -1
fi

bst_cnt=$(grep "Best Validation edge_classification" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation edge_classification"
    exit -1
fi

cnt=$(grep "Validation edge_classification" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Validation edge_classification"
    exit -1
fi

bst_cnt=$(grep "Best Test edge_regression" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test edge_regression"
    exit -1
fi

cnt=$(grep "Test edge_regression" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Test edge_regression"
    exit -1
fi

bst_cnt=$(grep "Best Validation edge_regression" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation edge_regression"
    exit -1
fi

cnt=$(grep "Validation edge_regression" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Validation edge_regression"
    exit -1
fi

bst_cnt=$(grep "Best Test link_prediction" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test link_prediction"
    exit -1
fi

cnt=$(grep "Test link_prediction" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Test link_prediction"
    exit -1
fi

bst_cnt=$(grep "Best Validation link_prediction" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation link_prediction"
    exit -1
fi

cnt=$(grep "Validation link_prediction" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Validation link_prediction"
    exit -1
fi

bst_cnt=$(grep "Best Test reconstruct_node_feat" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Test reconstruct_node_feat"
    exit -1
fi

cnt=$(grep "Test reconstruct_node_feat" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Test reconstruct_node_feat"
    exit -1
fi

bst_cnt=$(grep "Best Validation reconstruct_node_feat" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have Best Validation reconstruct_node_feat"
    exit -1
fi

cnt=$(grep "Validation reconstruct_node_feat" /tmp/train_log.txt | wc -l)
if test $cnt -lt $((1+$bst_cnt))
then
    echo "We use SageMaker task tracker, we should have Validation reconstruct_node_feat"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_mt/ | grep epoch | wc -l)
if test $cnt != 3
then
    echo "The number of save models $cnt is not equal to the specified topk 3"
    exit -1
fi

rm -fr /data/gsgnn_mt/
rm /tmp/train_log.txt

echo "**************[Multi-task with learnable embedding] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, with learnable node embedding, inference: full-graph, save model"
python3 -m graphstorm.run.gs_multi_task_learning --workspace $GS_HOME/training_scripts/gsgnn_mt  --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_ec_er_lp.yaml --save-model-path /data/gsgnn_mt/ --save-model-frequency 1000 --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True --use-node-embeddings True

error_and_exit $?

rm /tmp/train_log.txt
rm -frm /data/gsgnn_mt/

echo "**************[Multi-task] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, save model"
python3 -m graphstorm.run.gs_multi_task_learning --workspace $GS_HOME/training_scripts/gsgnn_mt  --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_ec_er_lp.yaml --save-model-path /data/gsgnn_mt/ --save-model-frequency 1000 --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True --use-mini-batch-infer False --save-embed-path /data/gsgnn_mt/emb/

error_and_exit $?

cnt=$(grep "save_embed_path: /data/gsgnn_mt/emb/" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_embed_path"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_mt/emb/ | wc -l)
cnt=$[cnt - 1]
if test $cnt != 2
then
    echo "The number of saved embs $cnt is not equal to 2 (for movie and user)."
fi

echo "**************[Multi-task] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch load from saved model and train"
python3 -m graphstorm.run.gs_multi_task_learning --workspace $GS_HOME/training_scripts/gsgnn_mt  --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_ec_er_lp.yaml --restore-model-path /data/gsgnn_mt/epoch-2/ --save-model-path /data/gsgnn_mt_2/ --save-model-frequency 1000 --logging-file /tmp/train_log.txt --logging-level debug

error_and_exit $?

cnt=$(ls -l /data/gsgnn_mt_2/ | grep epoch | wc -l)
if test $cnt != 3
then
    echo "The number of save models $cnt is not equal to the specified topk 3"
    exit -1
fi

echo "**************[Multi-task gen embedding] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, load from saved model"
python3 -m graphstorm.run.gs_gen_node_embedding --workspace $GS_HOME/training_scripts/gsgnn_mt/ --num-trainers $NUM_TRAINERS --use-mini-batch-infer false --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_ec_er_lp.yaml --save-embed-path /data/gsgnn_mt/save-emb/ --restore-model-path /data/gsgnn_mt/epoch-2/ --restore-model-layers embed,gnn --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True

error_and_exit $?

cnt=$(ls -l /data/gsgnn_mt/save-emb/ | wc -l)
cnt=$[cnt - 1]
if test $cnt != 2
then
    echo "The number of saved embs $cnt is not equal to 2 (for movie and user)."
fi

# Multi-task will save node embeddings of all the nodes.
python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_mt/emb/ --infer-embout /data/gsgnn_mt/save-emb/

rm -fr /tmp/train_log.txt

# Test inference for multi-task learning
echo "**************[Multi-task] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference only"
python3 -m graphstorm.run.gs_multi_task_learning --inference --workspace $GS_HOME/inference_scripts/mt_infer  --num-trainers $NUM_INFERs --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_ec_er_lp_only_infer.yaml --use-mini-batch-infer false  --save-embed-path /data/gsgnn_mt/infer-emb/ --restore-model-path /data/gsgnn_mt/epoch-2 --save-prediction-path /data/gsgnn_mt/prediction/ --logging-file /tmp/log.txt --preserve-input True --backend nccl

error_and_exit $?

cnt=$(ls -l /data/gsgnn_mt/infer-emb/ | wc -l)
cnt=$[cnt - 1]
if test $cnt != 3
then
    echo "The number of saved embs $cnt is not equal to 3 (for movie, user and the relations)."
fi

cnt=$(ls -l /data/gsgnn_mt/prediction | wc -l)
cnt=$[cnt - 1]
if test $cnt != 4
then
    echo "There are 4 prediction tasks, but got prediction results for $cnt"
    exit -1
fi

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_mt/emb/ --infer-embout /data/gsgnn_mt/infer-emb/

error_and_exit $?

rm -fr /data/gsgnn_mt/infer-emb/
rm -fr /data/gsgnn_mt/prediction/


echo "**************[Multi-task] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference only with mini-batch inference"
python3 -m graphstorm.run.gs_multi_task_learning --inference --workspace $GS_HOME/inference_scripts/mt_infer  --num-trainers $NUM_INFERs --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_ec_er_lp_only_infer.yaml --use-mini-batch-infer true  --save-embed-path /data/gsgnn_mt/infer-emb/ --restore-model-path /data/gsgnn_mt/epoch-2 --save-prediction-path /data/gsgnn_mt/prediction/ --logging-file /tmp/log.txt --preserve-input True --backend nccl

error_and_exit $?

cnt=$(ls -l /data/gsgnn_mt/infer-emb/ | wc -l)
cnt=$[cnt - 1]
if test $cnt != 3
then
    echo "The number of saved embs $cnt is not equal to 3 (for movie, user and the relations)."
fi

cnt=$(ls -l /data/gsgnn_mt/prediction | wc -l)
cnt=$[cnt - 1]
if test $cnt != 4
then
    echo "There are 4 prediction tasks, but got prediction results for $cnt"
    exit -1
fi

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_mt/emb/ --infer-embout /data/gsgnn_mt/infer-emb/

error_and_exit $?

rm -fr /data/gsgnn_mt/infer-emb/
rm -fr /data/gsgnn_mt/prediction/

echo "**************[Multi-task] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference with test"
python3 -m graphstorm.run.gs_multi_task_learning --inference --workspace $GS_HOME/inference_scripts/mt_infer  --num-trainers $NUM_INFERs --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_ec_er_lp_with_mask_infer.yaml --use-mini-batch-infer false  --save-embed-path /data/gsgnn_mt/infer-emb/ --restore-model-path /data/gsgnn_mt/epoch-2 --save-prediction-path /data/gsgnn_mt/prediction/ --logging-file /tmp/infer_log.txt --preserve-input True --backend nccl

error_and_exit $?

cnt=$(ls -l /data/gsgnn_mt/infer-emb/ | wc -l)
cnt=$[cnt - 2]
if test $cnt != 3
then
    echo "The number of saved embs $cnt is not equal to 3 (for movie, user and the relations)."
fi

cnt=$(ls -l /data/gsgnn_mt/prediction | wc -l)
cnt=$[cnt - 1]
if test $cnt != 4
then
    echo "There are 4 prediction tasks, but got prediction results for $cnt"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_mt/prediction/edge_classification-user_rating_movie-rate_class/user_rating_movie/ | grep predict | wc -l)
if test $cnt != $[$NUM_INFERs * 2]
then
    echo "The number of saved prediction result files is $cnt which does not equal to $NUM_INFERs (the number of inferers) * 2 as --preserve-input is True, for user,rating,movie rate edge classification"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_mt/prediction/edge_classification-user_rating_movie-rate_class/user_rating_movie/ | grep pred.predict | wc -l)
if test $cnt != $NUM_INFERs
then
    echo "The number of final prediction result files (parquet files) is $cnt, which must be $NUM_INFERs for user,rating,movie rate edge classification"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_mt/prediction/edge_regression-user_rating_movie-rate/user_rating_movie/ | grep predict | wc -l)
if test $cnt != $[$NUM_INFERs * 2]
then
    echo "The number of saved prediction result files is $cnt which does not equal to $NUM_INFERs (the number of inferers) * 2 as --preserve-input is True, for user,rating,movie rate edge regression"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_mt/prediction/edge_regression-user_rating_movie-rate/user_rating_movie/ | grep pred.predict | wc -l)
if test $cnt != $NUM_INFERs
then
    echo "The number of final prediction result files (parquet files) is $cnt, which must be $NUM_INFERs for user,rating,movie rate edge regression"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_mt/prediction/node_classification-movie-label/movie | grep predict- | wc -l)
if test $cnt != $[$NUM_INFERs * 2]
then
    echo "The number of saved prediction result files is $cnt which does not equal to $NUM_INFERs (the number of inferers) * 2 as --preserve-input is True, for movie node classification task 0"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_mt/prediction/node_classification-movie-label/movie | grep pred.predict | wc -l)
if test $cnt != $NUM_INFERs
then
    echo "The number of final prediction result files (parquet files) is $cnt, which must be $NUM_INFERs for movie node classification task 0"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_mt/prediction/node_classification-movie-label2/movie | grep predict- | wc -l)
if test $cnt != $[$NUM_INFERs * 2]
then
    echo "The number of saved prediction result files is $cnt which does not equal to $NUM_INFERs (the number of inferers) * 2 as --preserve-input is True, for movie node classification task 1"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_mt/prediction/node_classification-movie-label2/movie | grep pred.predict | wc -l)
if test $cnt != $NUM_INFERs
then
    echo "The number of final prediction result files (parquet files) is $cnt, which must be $NUM_INFERs for movie node classification task 1"
    exit -1
fi

# check prints
bst_cnt=$(grep "Best Test node_classification" /tmp/infer_log.txt | wc -l)
if test $bst_cnt != 2
then
    echo "We use SageMaker task tracker, the number of Best Test node_classification prints should be 2."
    exit -1
fi

cnt=$(grep "Test node_classification" /tmp/infer_log.txt | wc -l)
if test $cnt != 4
then
    echo "We use SageMaker task tracker, the number of Test node_classification should be 4."
    exit -1
fi

bst_cnt=$(grep "Best Validation node_classification" /tmp/infer_log.txt | wc -l)
if test $bst_cnt != 2
then
    echo "We use SageMaker task tracker, the number of Best Validation accuracy node_classification should be 4."
    exit -1
fi

cnt=$(grep "Validation node_classification" /tmp/infer_log.txt | wc -l)
if test $cnt != 4
then
    echo "We use SageMaker task tracker, the number of Validation node_classification should be 4."
    exit -1
fi

bst_cnt=$(grep "Best Test edge_classification" /tmp/infer_log.txt | wc -l)
if test $bst_cnt != 1
then
    echo "We use SageMaker task tracker, the number of Best Test edge_classification should be 1."
    exit -1
fi

cnt=$(grep "Test edge_classification" /tmp/infer_log.txt | wc -l)
if test $cnt != 2
then
    echo "We use SageMaker task tracker, the number of Test edge_classification should be 2."
    exit -1
fi

bst_cnt=$(grep "Best Validation edge_classification" /tmp/infer_log.txt | wc -l)
if test $bst_cnt != 1
then
    echo "We use SageMaker task tracker, the number of Best Validation edge_classification should be 1."
    exit -1
fi

cnt=$(grep "Validation edge_classification" /tmp/infer_log.txt | wc -l)
if test $cnt != 2
then
    echo "We use SageMaker task tracker, the number of  Validation edge_classification should be 2."
    exit -1
fi

bst_cnt=$(grep "Best Test edge_regression" /tmp/infer_log.txt | wc -l)
if test $bst_cnt != 1
then
    echo "We use SageMaker task tracker, the number of Best Test edge_regression should be 1."
    exit -1
fi

cnt=$(grep "Test edge_regression" /tmp/infer_log.txt | wc -l)
if test $cnt != 2
then
    echo "We use SageMaker task tracker, the number of Test edge_regression should be 2."
    exit -1
fi

bst_cnt=$(grep "Best Validation edge_regression" /tmp/infer_log.txt | wc -l)
if test $bst_cnt != 1
then
    echo "We use SageMaker task tracker, the number of Best Validation edge_regression should be 1."
    exit -1
fi

cnt=$(grep "Validation edge_regression" /tmp/infer_log.txt | wc -l)
if test $cnt != 2
then
    echo "We use SageMaker task tracker, the number of Validation edge_regression should be 2."
    exit -1
fi

bst_cnt=$(grep "Best Test link_prediction" /tmp/infer_log.txt | wc -l)
if test $bst_cnt != 1
then
    echo "We use SageMaker task tracker, the number of Best Test link_prediction should be 1."
    exit -1
fi

cnt=$(grep "Test link_prediction" /tmp/infer_log.txt | wc -l)
if test $cnt != 2
then
    echo "We use SageMaker task tracker, the number of Test link_prediction should be 2."
    exit -1
fi

bst_cnt=$(grep "Best Validation link_prediction" /tmp/infer_log.txt | wc -l)
if test $bst_cnt != 1
then
    echo "We use SageMaker task tracker, the number of Best Validation link_prediction should be 1."
    exit -1
fi

cnt=$(grep "Validation link_prediction" /tmp/infer_log.txt | wc -l)
if test $cnt != 2
then
    echo "We use SageMaker task tracker, the number of Validation link_prediction should be 2."
    exit -1
fi

bst_cnt=$(grep "Best Test reconstruct_node_feat" /tmp/infer_log.txt | wc -l)
if test $bst_cnt != 1
then
    echo "We use SageMaker task tracker, the number of Best Test reconstruct_node_feat should be 1."
    exit -1
fi

cnt=$(grep "Test reconstruct_node_feat" /tmp/infer_log.txt | wc -l)
if test $cnt != 2
then
    echo "We use SageMaker task tracker, the number of Test reconstruct_node_feat should be 2."
    exit -1
fi

bst_cnt=$(grep "Best Validation reconstruct_node_feat" /tmp/infer_log.txt | wc -l)
if test $bst_cnt != 1
then
    echo "We use SageMaker task tracker,the number of Best Validation reconstruct_node_feat should be 1."
    exit -1
fi

cnt=$(grep "Validation reconstruct_node_feat" /tmp/infer_log.txt | wc -l)
if test $cnt != 2
then
    echo "We use SageMaker task tracker, the number of Validation reconstruct_node_feat should be 2."
    exit -1
fi

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_mt/emb/ --infer-embout /data/gsgnn_mt/infer-emb/

error_and_exit $?

rm -fr /data/gsgnn_mt/
rm -fr /tmp/infer_log.txt

# Check the case when multi-task inference does not have any prediction tasks
echo "**************[Multi-task] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, with lp and reconstruct tasks only"
python3 -m graphstorm.run.gs_multi_task_learning --workspace $GS_HOME/training_scripts/gsgnn_mt  --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp_rnf.yaml --save-model-path /data/gsgnn_mt/ --save-model-frequency 1000 --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True

error_and_exit $?

echo "**************[Multi-task] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference only with lp and reconstruct tasks only"
python3 -m graphstorm.run.gs_multi_task_learning --inference --workspace $GS_HOME/inference_scripts/mt_infer  --num-trainers $NUM_INFERs --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp_no_prediction.yaml --use-mini-batch-infer false  --save-embed-path /data/gsgnn_mt/infer-emb/ --restore-model-path /data/gsgnn_mt/epoch-2 --save-prediction-path /data/gsgnn_mt/prediction/ --logging-file /tmp/log.txt --preserve-input True --backend nccl

error_and_exit $?

if [ -f /data/gsgnn_mt/prediction/ ]; then
    echo "The prediction result path should be empty"
    exit -1
fi
rm -fr /tmp/log.txt

echo "**************[Multi-task] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, save model"
python3 -m graphstorm.run.gs_multi_task_learning --workspace $GS_HOME/training_scripts/gsgnn_mt  --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_lp_norm.yaml --save-model-path /data/gsgnn_mt/ --save-model-frequency 1000 --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True --use-mini-batch-infer False --save-embed-path /data/gsgnn_mt/emb/

error_and_exit $?

cnt=$(grep "save_embed_path: /data/gsgnn_mt/emb/" /tmp/train_log.txt | wc -l)
if test $cnt -lt 1
then
    echo "We use SageMaker task tracker, we should have save_embed_path"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_mt/emb/ | wc -l)
cnt=$[cnt - 1]
if test $cnt != 3
then
    echo "The number of saved embs $cnt is not equal to 3. Should have two for movie and user and One for link-prediction-subtask normalized embedding."
fi

echo "**************[Multi-task] dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference with test"
python3 -m graphstorm.run.gs_multi_task_learning --inference --workspace $GS_HOME/inference_scripts/mt_infer  --num-trainers $NUM_INFERs --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_lp_norm_with_mask_infer.yaml --use-mini-batch-infer false  --save-embed-path /data/gsgnn_mt/infer-emb/ --restore-model-path /data/gsgnn_mt/epoch-2 --save-prediction-path /data/gsgnn_mt/prediction/ --logging-file /tmp/infer_log.txt --preserve-input True

error_and_exit $?

cnt=$(ls -l /data/gsgnn_mt/infer-emb/ | wc -l)
cnt=$[cnt - 2]
if test $cnt != 4
then
     echo "The number of saved embs $cnt is not equal to 3. Should have two for movie and user and One for link-prediction-subtask normalized embedding."
fi

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_mt/emb/ --infer-embout /data/gsgnn_mt/infer-emb/

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/check_infer.py --train-embout /data/gsgnn_mt/emb/link_prediction-user_rating_movie --infer-embout /data/gsgnn_mt/infer-emb/link_prediction-user_rating_movie

error_and_exit $?

rm -fr /data/gsgnn_mt/
rm -fr /tmp/train_log.txt
rm -fr /tmp/infer_log.txt
