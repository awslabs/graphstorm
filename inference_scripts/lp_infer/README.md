# Inference only script for link prediction.

## Movielens link prediction
You can follow training_scripts/gsgnn_lp/README 'Movielens link prediction' section to pretrain an GNN model. Then we can use the inference script here to do the inference.

You can follow the following commands to train a fresh model:
```
$ GS_HOME=/graph-storm
$ export PYTHONPATH=$GS_HOME/python/
$ wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
$ unzip ml-100k.zip
$ rm ml-100k.zip
$ python3 $GS_HOME/python/graphstorm/data/tools/preprocess_movielens.py \
    --input_path ml-100k --output_path movielen-data
$ rm -R ml-100k
$ export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/construct_graph.py --name movie-lens-100k\
    --filepath movielen-data \
    --output data \
    --dist_output movielen_100k_lp_train_val_1p_4t \
    --num_dataset_workers 10 \
    --hf_bert_model bert-base-uncased \
    --ntext_fields "movie:title" \
    --predict_etypes "user,rating,movie user,has-occupation,occupation" \
    --num_parts 1 \
    --num_trainers_per_machine 4 \
    --balance_train \
    --balance_edges \
    --generate_new_edge_split true \
    --device 0 \
    --undirected

$ python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_lp --num_trainers $NUM_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_lp.py --cf ml_lp.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie --num-gpus $NUM_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --save-model-path /data/gsgnn_lp_ml_dot/ --save-model-per-iter 0
```
The saved model will under /data/gsgnn_lp_ml_dot/-2/

### Do inference and also testing
You can do offline inference while providing a test set to do offline testing. (Please note that the dataset under /data/movielen_100k_lp_train_val_1p_4t/ has test_mask on test edges.)

```
$ python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/lp_infer --num_trainers $NUM_INFO_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 lp_infer_gnn.py --cf ml_lp_infer.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --num-gpus $NUM_INFO_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --save-embed-path /data/gsgnn_lp_ml_dot/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_dot/-2/"
```

### Do inference without doing testing
If you specify no_validation as True in the config file or pass it through commandline, the inference script will skip the testing part. If a dataset does not contain edges with test_mask, it will also skip testing.

```
$ python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/inference_scripts/lp_infer --num_trainers $NUM_INFO_TRAINERS --num_servers 1 --num_samplers 0 --part_config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip_config ip_list.txt --ssh_port 2222 "python3 lp_infer_gnn.py --cf ml_lp_infer.yaml --fanout '10,15' --n-layers 2 --mini-batch-infer false  --use-node-embeddings true --num-gpus $NUM_INFO_TRAINERS --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --save-embed-path /data/gsgnn_lp_ml_dot/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_dot/-2/ --no_validation True"
```
