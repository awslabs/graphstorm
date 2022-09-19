# Movielens node regression
Preparing movielens dataset for node regression. Case 1: Movie nodes use movie title as text feature and user nodes are featureless.
```
$ GS_HOME=/fsx-dev/xiangsx/home/workspace/graph-storm
$ export PYTHONPATH=$GS_HOME/python/
$ wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
$ unzip ml-100k.zip
$ rm ml-100k.zip
$ python3 $GS_HOME/python/graphstorm/data/tools/preprocess_movielens_builtin.py --filepath ./ --savepath movielen-data --max_sequence_length 64 --retain_original_features False --user_age_as_label true
$ python3 -u $GS_HOME/tools/partition_graph.py --dataset movie-lens-100k --filepath movielen-data --num_parts 1 --num_trainers_per_machine 4 --output movielen_100k_train_val_1p_4t --predict_ntypes movie,user
```

Case 2: Movie nodes use movie title as text feature and user nodes use occuption as text feature.
$ python3 $GS_HOME/python/graphstorm/data/tools/preprocess_movielens_builtin.py --filepath ./ --savepath movielen-data-utext --max_sequence_length 64 --retain_original_features True --user_text True --user_age_as_label true
$ python3 -u $GS_HOME/tools/partition_graph.py --dataset movie-lens-100k --filepath movielen-data-utext --num_parts 1 --num_trainers_per_machine 4 --output movielen_100k_utext_train_val_1p_4t --predict_ntypes movie,user
```

Training
```
DGL_HOME=/fsx-dev/xiangsx/home/workspace/dgl/dgl
python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_nr \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config movielen_100k_train_val_1p_4t/movie-lens-100k.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
    --ip_config ip_list.txt \
    "python3 gsgnn_nr_huggingface.py --cf ml_nr.yaml"
```

# Yelp node regression
Yelp dataset has multiple node types and edges. Here we perform node regression on business nodes, where we predict the rating of business. Here we treat the ratings as floating-point values.

## Preparation
You need to create a pre-processed the Yelp dataset before training. Following are the example CMDs to create such a dataset:

```
$ GS_HOME=/fsx-dev/xiangsx/home/workspace/graph-storm
$ export PYTHONPATH=$GS_HOME/python/
$ cd $GS_HOME/training_scripts/gsgnn_nc
$ aws s3 cp --recursive s3://search-m5-app-fsx-us-east-1-prod/FSxLustre20201016T182138Z/dzzhen/home/data/yelp_origin .
$ python3 $GS_HOME/python/graphstorm/data/tools/preprocess_yelp.py --input_path yelp_origin --output_path yelp_post --node_file_size 1000000
$ python3 $GS_HOME/tools/construct_graph.py --name yelp --filepath yelp_post --dist_output yelp_undirected_hf_emb_regress_1p_4t --num_dataset_workers 10 --hf_bert_model bert-base-uncased --ntext_fields "review:text" --nlabel_fields "business:stars" --ntask_types "business:regression" --predict_ntype business --num_parts 1 --balance_train --balance_edges --num_trainers_per_machine 4 --generate_new_split true --compute_bert_emb true --device 0 --remove_text_tokens true --undirected
```

The output file is the folder of `yelp_undirected_hf_emb_regress_1p_4t`. It contains a partitioned DGLGraph with a signle partition.

## Training
We can launch the training task.

```
$ DGL_HOME=/fsx-dev/xiangsx/home/workspace/dgl/dgl
$ python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nr --num_trainers 4 --num_servers 4 --num_samplers 0 --part_config yelp_undirected_hf_emb_regress_1p_4t/yelp.json --ip_config ip_list.txt "python3 gsgnn_nr_huggingface.py --cf yelp_nr.yaml"

```
