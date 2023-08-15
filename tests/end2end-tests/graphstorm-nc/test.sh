#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=1
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_np

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

echo "Test GraphStorm node classification"

date

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, no test_set"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_notest_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml | tee train_log.txt

error_and_exit ${PIPESTATUS[0]}

bst_cnt=$(grep "Best Test accuracy: N/A" train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "Test set is empty we should have Best Test accuracy: N/A"
    exit -1
fi

mkdir -p /tmp/ML_np_profile

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, with profiling"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --profile-path /tmp/ML_np_profile

error_and_exit $?

cnt=$(ls /tmp/ML_np_profile/*.csv | wc -l)
if test $cnt -lt 1
then
    echo "Cannot find the profiling files."
    exit -1
fi

rm -R /tmp/ML_np_profile

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --use-mini-batch-infer false

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, mlp layer between GNN layer: 1"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --use-mini-batch-infer false --num-ffn-layers-in-gnn 1 --save-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model

error_and_exit $?

python3 -m graphstorm.run.gs_node_classification --inference --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --restore-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model/epoch-1/

error_and_exit $?

rm -R ./models/movielen_100k/train_val/movielen_100k_ngnn_model

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, mlp layer in input layer: 1"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --use-mini-batch-infer false --num-ffn-layers-in-input 1 --save-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model

error_and_exit $?

python3 -m graphstorm.run.gs_node_classification --inference --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --restore-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model/epoch-1/

error_and_exit $?

rm -R ./models/movielen_100k/train_val/movielen_100k_ngnn_model

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --use-node-embeddings true --use-mini-batch-infer false

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 2, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false   --save-model-path ./models/movielen_100k/train_val/movielen_100k_utext_train_val_1p_4t_model --save-model-frequency 1000

error_and_exit $?

echo "**************restart training from iteration 1 of the previous training"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --fanout '10,15' --num-layers 2 --restore-model-path ./models/movielen_100k/train_val/movielen_100k_utext_train_val_1p_4t_model/epoch-1

error_and_exit $?

## load emb from previous run and check its shape
python3 $GS_HOME/tests/end2end-tests/graphstorm-nc/check_emb.py --emb-path "./models/movielen_100k/train_val/movielen_100k_utext_train_val_1p_4t_model/epoch-1/" --graph-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ntypes "movie user" --emb-size 128

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/graphstorm-nc/check_emb.py --emb-path "./models/movielen_100k/train_val/movielen_100k_utext_train_val_1p_4t_model/epoch-2/" --graph-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ntypes "movie user" --emb-size 128

error_and_exit $?


echo "**************dataset: MovieLens, RGAT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --model-encoder-type rgat

error_and_exit $?

echo "**************dataset: MovieLens, RGAT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, num-heads: 8"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --model-encoder-type rgat --num-heads 8

error_and_exit $?

echo "**************dataset: MovieLens, RGAT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --model-encoder-type rgat --use-mini-batch-infer false

error_and_exit $?

echo "**************dataset: MovieLens, RGAT layer: 2, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --model-encoder-type rgat --fanout '5,10' --num-layers 2 --use-mini-batch-infer false

error_and_exit $?

rm -Rf /data/movielen_100k_multi_label_nc
cp -R /data/movielen_100k_train_val_1p_4t /data/movielen_100k_multi_label_nc
python3 $GS_HOME/tests/end2end-tests/data_gen/gen_multilabel.py --path /data/movielen_100k_multi_label_nc --node_class 1 --field label

echo "**************dataset: multilabel MovieLens, RGCN layer: 1, node feat: generated feature, inference: mini-batch, save emb"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_nc/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --save-embed-path ./model/ml-emb/ --num-epochs 3 --multilabel true --num-classes 6 --node-feat-name movie:title user:feat

error_and_exit $?

echo "**************dataset: multilabel MovieLens with weight, RGCN layer: 1, node feat: generated feature, inference: full graph, save emb"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_nc/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --save-embed-path ./model/ml-emb/ --num-epochs 3 --multilabel true --num-classes 6 --node-feat-name movie:title user:feat --use-mini-batch-infer false --multilabel-weights 0.2,0.2,0.1,0.1,0.2,0.2

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 2, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, imbalance-class"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --imbalance-class-weights 1,1,1,1,2,1,1,1,1,2,1,1,1,1,2,1,1,1,1

error_and_exit $?

rm -Rf /data/movielen_100k_multi_node_feat_nc
cp -R /data/movielen_100k_train_val_1p_4t /data/movielen_100k_multi_node_feat_nc
python3 $GS_HOME/tests/end2end-tests/data_gen/gen_multi_feat_nc.py --path /data/movielen_100k_multi_node_feat_nc

echo "**************dataset: multi-feature MovieLens, RGCN layer: 1, node feat: generated feature, inference: mini-batch"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_node_feat_nc/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --num-epochs 3 --node-feat-name user:feat0 movie:title

error_and_exit $?

rm -Rf /data/movielen_100k_multi_feat_nc
cp -R /data/movielen_100k_train_val_1p_4t /data/movielen_100k_multi_feat_nc
# generate a dataset with user and movie have multiple node features
python3 $GS_HOME/tests/end2end-tests/data_gen/gen_multi_feat_nc.py --path /data/movielen_100k_multi_feat_nc --multi_feats=True

echo "**************dataset: multi-feature MovieLens, RGCN layer: 1, node feat: generated feature, inference: mini-batch"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_feat_nc/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --num-epochs 3  --node-feat-name movie:title user:feat0,feat1

error_and_exit $?

date

echo 'Done'