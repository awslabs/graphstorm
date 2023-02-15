# MOVIE-LENs Edge Classification Example
Movie-lens edge classification example serves as the simplest GSGNN example for edge classification.

## Preparation
You need to create a pre-processed movie-lens edge classification dataset before training. Following are the example CMDs to create such a dataset:
```
$ GS_HOME=~/graphstorm/
$ export PYTHONPATH=$GS_HOME/python/
$ wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
$ unzip ml-100k.zip
$ rm ml-100k.zip
$ python3 $GS_HOME/python/graphstorm/data/tools/preprocess_movielens.py \
    --input_path ml-100k --output_path movielen-data-ec

$ rm -R ml-100k
$ python3 /$GS_HOME/tools/construct_graph.py --name movie-lens-100k \
    --undirected \
    --filepath movielen-data-ec \
    --output movielen-data-ec-graph \
    --dist_output movielen_100k_ec_1p_4t \
    --elabel_fields "user,rating,movie:rate" \
    --predict_etypes "user,rating,movie" \
    --etask_types "user,rating,movie:classification" \
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

The output file is movielen_100k_ec_1p_4t/. It contains a partitioned DGLGraph with a signle partition.

## Training
After copying the movielen_100k_ec_1p_4t folder into current location, We can launch the training task.

```
$ DGL_HOME=~/workspace/dgl/dgl
$ python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_ec --num_trainers 4 --num_servers 4 --num_samplers 0 --part_config movielen_100k_ec_1p_4t/movie-lens-100k.json --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" --ip_config ip_list.txt --ssh_port 2222 "python3 gsgnn_ec.py --cf ml_ec.yaml"
```

