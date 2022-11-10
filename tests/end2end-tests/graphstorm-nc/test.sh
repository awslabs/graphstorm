#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=1
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_nc

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

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_nc_huggingface.py --cf ml_nc.yaml --train-nodes 0 --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json"

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_nc_huggingface.py --cf ml_nc.yaml --train-nodes 0 --mini-batch-infer false --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json"

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_nc_huggingface.py --cf ml_nc.yaml --train-nodes 0 --use-node-embeddings true --mini-batch-infer false --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json"

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 2, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_nc_huggingface.py --cf ml_nc.yaml --train-nodes 0 --fanout '10,15' --n-layers 2 --mini-batch-infer false --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --save-model-path ./models/movielen_100k/train_val/movielen_100k_utext_train_val_1p_4t_model --save-model-per-iters 1000"

error_and_exit $?

echo "**************restart training from iteration 1 of the previous training"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_nc_huggingface.py --cf ml_nc.yaml --train-nodes 0 --fanout '10,15' --n-layers 2 --restore-model-path ./models/movielen_100k/train_val/movielen_100k_utext_train_val_1p_4t_model/epoch-1"

error_and_exit $?

## load emb from previous run and check its shape
python3 $GS_HOME/tests/end2end-tests/graphstorm-nc/check_emb.py --emb-path "./models/movielen_100k/train_val/movielen_100k_utext_train_val_1p_4t_model/epoch-1/" --graph-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ntypes "movie user" --emb-size 128

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/graphstorm-nc/check_emb.py --emb-path "./models/movielen_100k/train_val/movielen_100k_utext_train_val_1p_4t_model/epoch-2/" --graph-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ntypes "movie user" --emb-size 128

error_and_exit $?


#echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, eval-metric: precision_recall accuracy f1_score per_class_f1_score"
#python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_nc_huggingface.py --cf ml_nc.yaml --train-nodes 0 --num-gpus $NUM_TRAINERS --eval-metric precision_recall accuracy f1_score per_class_f1_score --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json"

#error_and_exit $?

echo "**************dataset: MovieLens, RGAT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_nc_huggingface.py --cf ml_nc.yaml --model-encoder-type rgat --train-nodes 0 --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json"

error_and_exit $?

echo "**************dataset: MovieLens, RGAT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, n-heads: 8"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc/ --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_nc_huggingface.py --cf ml_nc.yaml --model-encoder-type rgat --train-nodes 0 --num-gpus $NUM_TRAINERS --n-heads 8 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json"

error_and_exit $?

echo "**************dataset: MovieLens, RGAT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_nc_huggingface.py --cf ml_nc.yaml --model-encoder-type rgat --train-nodes 0 --mini-batch-infer false --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json"

error_and_exit $?

echo "**************dataset: MovieLens, RGAT layer: 2, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_nc_huggingface.py --cf ml_nc.yaml --model-encoder-type rgat --train-nodes 0 --fanout '5,10' --n-layers 2 --mini-batch-infer false --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json"

error_and_exit $?

echo "**************dataset: Generated multilabel NC test, RGCN layer: 1, node feat: generated feature, inference: mini-batch, save emb"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_multilabel_nc_1p_4t/multilabel-nc-test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_pure_gnn_nc.py --cf ml_nc.yaml --save-embeds-path ./model/ml-emb/ --n-epochs 3 --graph-name multilabel-nc-test --part-config test_multilabel_nc_1p_4t/multilabel-nc-test.json --label-field label --predict-ntype ntype1 --multilabel true --num-classes 6 --feat-name feat"

error_and_exit $?

echo "**************dataset: Generated multilabel NC test, RGCN layer: 1, node feat: generated feature, inference: full graph, save emb"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_multilabel_nc_1p_4t/multilabel-nc-test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_pure_gnn_nc.py --cf ml_nc.yaml --save-embeds-path ./model/ml-emb/ --n-epochs 3 --graph-name multilabel-nc-test --part-config test_multilabel_nc_1p_4t/multilabel-nc-test.json --label-field label --predict-ntype ntype1 --multilabel true --num-classes 6 --feat-name feat --mini-batch-infer false"

error_and_exit $?

echo "**************dataset: Generated multilabel NC test with weight, RGCN layer: 1, node feat: generated feature, inference: full graph, save emb"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_multilabel_nc_1p_4t/multilabel-nc-test.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_pure_gnn_nc.py --cf ml_nc.yaml --save-embeds-path ./model/ml-emb/ --n-epochs 3 --graph-name multilabel-nc-test --part-config test_multilabel_nc_1p_4t/multilabel-nc-test.json --label-field label --predict-ntype ntype1 --multilabel true --num-classes 6 --feat-name feat --mini-batch-infer false --multilabel-weights 0.2,0.2,0.1,0.1,0.2,0.2"

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 2, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, imbalance-class"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_nc_huggingface.py --cf ml_nc.yaml --train-nodes 0 --fanout '10,15' --n-layers 2 --mini-batch-infer false --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --imbalance-class-weights 1,1,1,1,2,1,1,1,1,2,1,1,1,1,2,1,1,1,1"

error_and_exit $?

echo "**************dataset: Generated multi feat NC test, RGCN layer: 1, node feat: generated feature, inference: mini-batch, save emb"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_multi_feat_nc_4t/multi-feat-nc-test.json --ip_config ip_list.txt --ssh_port 2222 'python3 gsgnn_pure_gnn_nc.py --cf ml_nc.yaml --save-embeds-path ./model/ml-emb/ --n-epochs 3 --graph-name multi-feat-nc-test --part-config test_multi_feat_nc_4t/multi-feat-nc-test.json --label-field label --predict-ntype ntype1 --num-classes 6 --feat-name "ntype0:feat0 ntype1:feat1"'

error_and_exit $?

echo "**************dataset: Generated multi feat NC test, RGCN layer: 1, node feat: generated feature, inference: full graph, save emb"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_multi_feat_nc_4t/multi-feat-nc-test.json --ip_config ip_list.txt --ssh_port 2222 'python3 gsgnn_pure_gnn_nc.py --cf ml_nc.yaml --save-embeds-path ./model/ml-emb/ --n-epochs 3 --graph-name multi-feat-nc-test --part-config test_multi_feat_nc_4t/multi-feat-nc-test.json --label-field label --predict-ntype ntype1 --num-classes 6 --feat-name "ntype0:feat0 ntype1:feat1" --mini-batch-infer false'

error_and_exit $?

echo "**************dataset: Generated multi feat same name NC test, RGCN layer: 1, node feat: generated feature, inference: mini-batch, save emb"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_multi_feat_same_name_nc_4t/multi-feat-sn-nc-test.json --ip_config ip_list.txt --ssh_port 2222 'python3 gsgnn_pure_gnn_nc.py --cf ml_nc.yaml --save-embeds-path ./model/ml-emb/ --n-epochs 3 --graph-name multi-feat-sn-nc-test --part-config test_multi_feat_same_name_nc_4t/multi-feat-sn-nc-test.json --label-field label --predict-ntype ntype1 --num-classes 6 --feat-name "feat0"'

error_and_exit $?

echo "**************dataset: Generated multi feat same name NC test, RGCN layer: 1, node feat: generated feature, inference: full graph, save emb"
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/test_multi_feat_same_name_nc_4t/multi-feat-sn-nc-test.json --ip_config ip_list.txt --ssh_port 2222 'python3 gsgnn_pure_gnn_nc.py --cf ml_nc.yaml --save-embeds-path ./model/ml-emb/ --n-epochs 3 --graph-name multi-feat-sn-nc-test --part-config test_multi_feat_same_name_nc_4t/multi-feat-sn-nc-test.json --label-field label --predict-ntype ntype1 --num-classes 6 --feat-name "feat0" --mini-batch-infer false'

error_and_exit $?
