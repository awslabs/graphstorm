# Construct a graph from the standard input.
`tools/construct_graph.py` is a generic script for converting graph data in the GSGNN input format to the DGL format.

The example below constructs a DGL graph for a MovieLens dataset, which is stored with the standard format. Users can choose whether to load pre-processing node features or computes BERT embeddings.
After constructing the graph, it partitions the graph for
distributed training. It stores the constructed DGL graph in the folder of `data` and the partitioned results
in the folder of `ml_1p_4t`.

Example to compute embeddings:

```
python3 tools/construct_graph.py --name movielens --filepath /tmp/ml_post --output data --dist_output ml_1p_4t --num_dataset_workers 10 \
			--hf_bert_model bert-base-uncased --ntext_fields "movie:title" --nlabel_fields "movie:genre" --predict_ntype movie \
			--num_parts 1 --balance_train --balance_edges --num_trainers_per_machine 4 \
			--generate_new_split true --compute_bert_emb true --device 0 --remove_text_tokens true
```

Example to load node features:
```
python3 tools/construct_graph.py --name movielens --filepath /tmp/ml_post --output data --dist_output ml_1p_4t --num_dataset_workers 10 \
			 --ntext_fields "movie:title" --nlabel_fields "movie:genre" --predict_ntype movie \
			--num_parts 1 --balance_train --balance_edges --num_trainers_per_machine 4 \
			--generate_new_split true --compute_bert_emb true --device 0 --remove_text_tokens true --feat_format hdf5
```


# Partition a graph for a link prediction task

Partition a graph for a semantic matching task

filepath is the graph path
Specify num_trainers_per_machine larger than 1
num_parts is the number of machines number of partitions
output is the name of the partioned output folder
```
python3 partition_graph_lp.py --dataset query_asin_match --filepath qa_data_graph_v_1/ --num_parts 8 --num_trainers_per_machine 8 --output qa_train_v01_8p
```

# Partition paper100m
```
python3 /fsx-dev/xiangsx/home/workspace/graph-storm/tools/partition_graph.py --dataset ogbn-papers100m --filepath ./paper100m-processed-512/ --num_parts 8 --predict_ntypes "node" --balance_train --num_trainers_per_machine 8 --output /fsx-dev/xiangsx/home/workspace/graph-storm/training_scripts/gsgnn_nc/ogbn_papers100m_nc_8p_8t/
```

# Test the construct graph

## for edge class

```
/opt/conda/bin/python3 tools/construct_graph.py --name test --filepath  ~/graph-storm/python/graphstorm/data/test/data/edge_class/ --output data_test_ec --dist_output test_ec_1p_4t --num_dataset_workers 10 \
			--hf_bert_model bert-base-uncased --ntext_fields "node:text" --elabel_fields "node,r0,item:label" --predict_etype node,r0,item \
			--num_parts 1 --balance_train --balance_edges --num_trainers_per_machine 4 \
			 --device 0 --split_etypes node,r0,item
```

Example to load edge features and precomputed edge embeddings:
```
/opt/conda/bin/python3 ~/graph-storm/tools/construct_graph.py --name test --filepath  ~/graph-storm/python/graphstorm/data/test/data/edge_class_with_features_and_ext_feats/ --output data_test_ec --dist_output test_ec_1p_4t --num_dataset_workers 10 \
			--hf_bert_model bert-base-uncased --ntext_fields "node:text" --elabel_fields "node,r0,item:label" --predict_etype node,r0,item \
			--efeat_format npy --num_parts 1 --balance_train --balance_edges --num_trainers_per_machine 4 \
			 --device 0 --split_etypes node,r0,item
```
Example to load edge features:
```python3 tools/construct_graph.py \
	--name edge-feature-toy \
	--filepath python/graphstorm/data/test/data/edge_class_with_features \
	--output data/single \
	--dist_output data/dist \
	--num_parts 1 \
	--balance_train \
	--balance_edges \
	--num_trainers_per_machine 4 \
	--vocab /fsx-dev/vocab/bert_asin.model \
	--ntext_fields "item:id node:text" \
	--efeat_fields "node,r0,item:f1,f2" \
	--num_dataset_workers 2 \
	--undirected
```

## for node class with node features

```
/opt/conda/bin/python3 ~/graph-storm/tools/construct_graph.py --name test --filepath  ~/graph-storm/python/graphstorm/data/test/data/node_class_with_nfeats/ --output data_test_ec --dist_output test_ec_1p_4t --num_dataset_workers 10 \
			--hf_bert_model bert-base-uncased --ntext_fields "node:text" --nlabel_fields "node:label" --predict_ntype node \
			--nfeat_fields "node:cat" --num_parts 1 --balance_train --balance_edges --split_ntypes node --num_trainers_per_machine 4 \
			 --device 0

```

# Distributed data prcoessing
## Run data pre-processing for movielens
In this example we have only one machine. It runs two separate processes to do data processing
```
rm -fr ml-output

python3 launch_dist_process.py --num_workers 2 --ip_config ip_list.txt --workspace /fsx-dev/xiangsx/home/workspace/graph-storm/tools '/opt/conda/bin/python3 preprocess_dist_graph.py --name ml --filepath /fsx-dev/xiangsx/home/workspace/graph-storm/tools/ml-json --output ml-output --hf_bert_model "bert-base-uncased" --ntext_fields "movie:title" --nlabel_fields "movie:genre" --predict_ntype "movie" --ntask_types "movie:classify"  --generate_new_split true --ntypes "movie occupation user" --etypes "user;rating;movie user;has-occupation;occupation" --undirected'
```

## Run data pre-processing for yelp
python3 launch_dist_process.py --num_workers 4 --ip_config ip_list.txt --workspace /fsx-dev/xiangsx/home/workspace/graph-storm/tools '/opt/conda/bin/python3 preprocess_dist_graph.py --name yelp --filepath /data/graph-storm/dataset/yelp/yelp/ --output yelp-output --hf_bert_model "bert-base-uncased" --ntext_fields "review:text" --nlabel_fields "business:stars" --predict_ntype "business" --ntask_types "business:regression"  --generate_new_split true --ntypes "business category city review user" --etypes "business;incategory;category business;in;city review;on;business user;friendship;user user;write;review"'
