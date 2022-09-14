# Movielens edge regression
Preparing movielens dataset for edge regression. Movie nodes use movie title as text feature and user nodes are featureless.

```
$ M5GNN_HOME=/fsx-dev/xiangsx/home/workspace/m5-gnn
$ export PYTHONPATH=$M5GNN_HOME/python/
$ wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
$ unzip ml-100k.zip
$ rm ml-100k.zip
$ python3 /m5-gnn/python/m5gnn/data/tools/preprocess_movielens.py \
    --input_path ml-100k --output_path movielen-data-er

$ rm -R ml-100k
$ python3 /m5-gnn/tools/construct_graph.py --name movie-lens-100k \
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

Training
```
DGL_HOME=/fsx-dev/xiangsx/home/workspace/dgl/dgl
python3 $DGL_HOME/tools/launch.py \
    --ssh_port 2222 \
    --workspace $M5GNN_HOME/training_scripts/m5gnn_er \
    --num_trainers 4 --num_servers 1 --num_samplers 0 \
    --part_config movielen_100k_er_1p_4t/movie-lens-100k.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
    --ip_config ip_list.txt \
    "python3 m5gnn_er_huggingface.py --cf ml_er.yaml"
```