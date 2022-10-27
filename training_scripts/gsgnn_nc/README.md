# ARXIV Node Classification Example
Arxiv node classification example serves as the simplest GSGNN example. It shows how to use yaml files to choose difference configurations of GSGNN.

## Preparation
You need to create a pre-processed arxiv node classification dataset before training. Following are the example CMDs to create such a dataset:

```
$ GS_HOME=/graph-storm
$ export PYTHONPATH=$GS_HOME/python/
$ cd $GS_HOME/training_scripts/gsgnn_nc
$ aws s3 cp --recursive s3://graphstorm-example/arxiv/ogbn-arxiv-raw/ ogbn-arxiv-raw/
$ python3 $GS_HOME/tools/gen_ogbn_dataset.py --filepath ogbn-arxiv-raw/ --savepath ogbn-arxiv/
$ python3 -u $GS_HOME/tools/partition_graph.py --dataset ogbn-arxiv --filepath ogbn-arxiv/ --num_parts 1 --num_trainers_per_machine 4 --output ogb_arxiv_nc_train_val_1p_4t
```

The output file is ogb_arxiv_nc_train_val_1p_4t/. It contains a partitioned DGLGraph with a signle partition.

## Training
After copying the ogb_arxiv_nc_train_val_1p_4t folder into current location (under arxiv_lp), We can launch the training task.

```
$ DGL_HOME=/fsx-dev/xiangsx/home/workspace/dgl/dgl
$ python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers 4 --num_servers 4 --num_samplers 0 --part_config ogb_arxiv_nc_train_val_1p_4t/ogbn-arxiv.json --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" --ip_config ip_list.txt "python3 gsgnn_nc_huggingface.py --cf arxiv_nc_hf.yaml"
```

## Test scripts:
train+validation+mixed-precision-O2+save-model+save-embeds
```
python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_nc \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config ogb_arxiv_nc_train_val_1p_4t/ogbn-arxiv.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
    --ip_config ip_list.txt \
    "python3 gsgnn_nc_huggingface.py --cf arxiv_nc_hf.yaml"
```

train+validation+mixed-precision-O1
```
python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_nc \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config ogb_arxiv_nc_train_val_1p_4t/ogbn-arxiv.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
    --ip_config ip_list.txt \
    "python3 gsgnn_nc_huggingface.py --cf arxiv_nc_hf.yaml --mp-opt-level O1 --save-model-path none --save-embeds-path none --negative-sampler uniform"
```

train+validation+mixed-precision-O1-full-graph-infer
```
python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_nc \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config ogb_arxiv_nc_train_val_1p_4t/ogbn-arxiv.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
    --ip_config ip_list.txt \
    "python3 gsgnn_nc_huggingface.py --cf arxiv_nc_hf.yaml --mp-opt-level O1 --save-model-path none --save-embeds-path none --save-model-per-iters 0 --mini-batch-infer false"
```

train+validation+bert-cache
```
python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_nc \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config ogb_arxiv_nc_train_val_1p_4t/ogbn-arxiv.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
    --ip_config ip_list.txt \
    "python3 gsgnn_nc_huggingface.py --cf arxiv_nc_hf.yaml --use-bert-cache true --refresh-cache true --mixed-precision false --save-model-path none --save-embeds-path none --negative-sampler localuniform"
```

 * arxiv_nc_nemb.yaml: train+validation+mixed-precision-O2+save-model+save-embeds+user-node-embedding

 ## None-Bert Training
 Generate a graph data without g.nodes['node'].data['text_idx']
```
$ python3 $GS_HOME/tools/gen_ogbn_dataset.py --filepath ogbn-arxiv-raw/ --savepath ogb-arxiv-origin/ --retain_original_features True
$ python3 -u $GS_HOME/tools/partition_graph.py --dataset ogbn-arxiv --filepath ogb-arxiv-origin/ --num_parts 1 --num_trainers_per_machine 4 --output ogb_arxiv_origin_1p_4t
```

```
$ DGL_HOME=/fsx-dev/xiangsx/home/workspace/dgl/dgl
$ python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_nc \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config ogb_arxiv_origin_1p_4t/ogbn-arxiv.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
    --ip_config ip_list.txt \
    "python3 gsgnn_pure_gnn_nc.py --cf arxiv_nc_hf.yaml --part-config 'ogb_arxiv_origin_1p_4t/ogbn-arxiv.json' --fanout '10,15' --n-layers 2 --save-model-path './models/ogb_arxiv/train_val/ogb_arxiv_origin_1p_4t_model' --save-embeds-path './models/ogb_arxiv/train_val/ogb_arxiv_origin_1p_4t_emb'"
```

 ## Use Huggingface Bert
 ```
$ DGL_HOME=/fsx-dev/xiangsx/home/workspace/dgl/dgl
$ python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_nc \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config ogb_arxiv_nc_train_val_1p_4t/ogbn-arxiv.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
    --ip_config ip_list.txt \
    "python3 gsgnn_nc_huggingface.py --cf gsgnn_nc_hf.yaml"
```

### Use a finetuned hugging face bert model

 ```
nohup python3 -u ~/dgl/tools/launch.py --workspace ~/graphstorm/training_scripts/gsgnn_nc --num_trainers 8 --num_servers 1 --num_samplers 0 --part_config /fsx-dev/ivasilei/home/ogbn_text_graph_data/ogbn-arxiv-graph-512-scibert-nc-1p/ogbn-arxiv.json --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" --ip_config ip_list.txt "python3 gsgnn_nc_huggingface.py --cf arxiv_nc_hf_ft.yaml" > t0nmpnp.out 2> t0nmpnp.err &
```

# Movielens node classification
Preparing movielens dataset for node classification. Case 1: Movie nodes use movie title as text feature and user nodes are featureless.
```
$ GS_HOME=/graph-storm
$ export PYTHONPATH=$GS_HOME/python/
$ wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
$ unzip ml-100k.zip
$ rm ml-100k.zip
$ python3 $GS_HOME/python/graphstorm/data/tools/preprocess_movielens_builtin.py --filepath ./ --savepath movielen-data --max_sequence_length 64 --retain_original_features False
$ python3 -u $GS_HOME/tools/partition_graph.py --dataset movie-lens-100k --filepath movielen-data --num_parts 1 --num_trainers_per_machine 4 --output movielen_100k_train_val_1p_4t
```

Case 2: Movie nodes use movie title as text feature and user nodes use occuption as text feature.
$ python3 $GS_HOME/python/graphstorm/data/tools/preprocess_movielens_builtin.py --filepath ./ --savepath movielen-data-utext --max_sequence_length 64 --retain_original_features True --user_text True
$ python3 -u $GS_HOME/tools/partition_graph.py --dataset movie-lens-100k --filepath movielen-data-utext --num_parts 1 --num_trainers_per_machine 4 --output movielen_100k_utext_train_val_1p_4t
```

Training
```
DGL_HOME=/fsx-dev/xiangsx/home/workspace/dgl/dgl
python3 $DGL_HOME/tools/launch.py \
    --workspace $GS_HOME/training_scripts/gsgnn_nc \
    --num_trainers 4 --num_servers 4 --num_samplers 0 \
    --part_config movielen_100k_train_val_1p_4t/movie-lens-100k.json \
    --extra_envs "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH" \
    --ip_config ip_list.txt \
    "python3 gsgnn_nc_huggingface.py --cf ml_nc.yaml"
```

# Yelp Node Classification Example
Yelp dataset has multiple node types and edges. Here we perform node classification on business nodes, where we predict the rating of business.

## Preparation
You need to create a pre-processed the Yelp dataset before training. Following are the example CMDs to create such a dataset:

```
$ GS_HOME=/graph-storm
$ export PYTHONPATH=$GS_HOME/python/
$ cd $GS_HOME/training_scripts/gsgnn_nc
$ aws s3 cp --recursive s3://search-m5-app-fsx-us-east-1-prod/FSxLustre20201016T182138Z/dzzhen/home/data/yelp_origin .
$ python3 $GS_HOME/python/graphstorm/data/tools/preprocess_yelp.py --input_path yelp_origin --output_path yelp_post --node_file_size 1000000
$ python3 $GS_HOME/tools/construct_graph.py --name yelp --filepath yelp_post --dist_output yelp_undirected_hf_emb_1p_4t --num_dataset_workers 10 --hf_bert_model bert-base-uncased --ntext_fields "review:text" --nlabel_fields "business:stars" --predict_ntype business --ntask_types "business:classify" --num_parts 1 --balance_train --balance_edges --num_trainers_per_machine 4 --generate_new_split true --compute_bert_emb true --device 0 --remove_text_tokens true --undirected
```

The output file is the folder of `yelp_undirected_hf_emb_1p_4t`. It contains a partitioned DGLGraph with a signle partition.

## Training
We can launch the training task.

```
$ DGL_HOME=/fsx-dev/xiangsx/home/workspace/dgl/dgl
$ python3 $DGL_HOME/tools/launch.py --workspace $GS_HOME/training_scripts/gsgnn_nc --num_trainers 4 --num_servers 1 --num_samplers 0 --part_config yelp_undirected_hf_emb_1p_4t/yelp.json --ip_config ip_list.txt "python3 gsgnn_nc_huggingface.py --cf yelp_nc.yaml"
```
