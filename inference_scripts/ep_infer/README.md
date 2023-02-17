# Inference only script for edge classification/regression

## Movielens edge regression
You can follow training_scripts/gsgnn_er/README 'Movielens edge regression' section to pretrain an GNN model. Then we can use the inference script here to do the inference.

You can follow the following commands to build a movielens edge regression dataset and train a fresh model:

```
$ GS_HOME=/graph-storm
$ export PYTHONPATH=$GS_HOME/python/
$ wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
$ unzip ml-100k.zip
$ rm ml-100k.zip
$ python3 $GS_HOME/python/graphstorm/data/tools/preprocess_movielens.py \
    --input_path ml-100k --output_path movielen-data-er

$ rm -R ml-100k
$ python3 $GS_HOME/tools/construct_graph.py --name movie-lens-100k \
	--undirected \
    --filepath movielen-data-er \
    --output data \
    --dist_output movielen_100k_er_1p_4t \
    --elabel_fields "user,rating,movie:rate" \
    --predict_etypes "user,rating,movie" \
    --etask_types "user,rating,movie:regression" \
    --num_dataset_workers 10 \
    --hf_bert_model bert-base-uncased \
    --ntext_fields "movie:title" \
    --num_parts 1 \
    --num_trainers_per_machine 4 \
    --balance_train \
    --balance_edges \
    --generate_new_edge_split true \
    --device 0
```

```
DGL_HOME=/fsx-dev/xiangsx/home/workspace/dgl/dgl
python3 $DGL_HOME/tools/launch.py \
    --ssh_port 2222 \
    --workspace $GS_HOME/training_scripts/gsgnn_er \
    --num_trainers 4 --num_servers 1 --num_samplers 0 \
    --part_config movielen_100k_er_1p_4t/movie-lens-100k.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
    --ip_config ip_list.txt \
    "python3 gsgnn_er.py --cf ml_er.yaml  --save-model-path /data/gsgnn_er/ --save-model-per-iter 0"
```

## Do Inference
```
python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/ep_infer --num_trainers $NUM_INFO_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 ep_infer_gnn.py --cf ml_er_infer.yaml --num-gpus $NUM_INFO_TRAINERS --part-config /data/movielen_100k_er_1p_4t/movie-lens-100k.json --mini-batch-infer false --save-embed-path /data/gsgnn_er/infer-emb/ --restore-model-path /data/gsgnn_er/-2/"
```

