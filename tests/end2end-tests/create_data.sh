GS_HOME=$(pwd)
mkdir -p /data
cd /data
wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
rm ml-100k.zip

python3 /$GS_HOME/python/graphstorm/data/tools/preprocess_movielens.py \
    --input_path ml-100k --output_path movielen-data
rm -fr ml-json

python3 /$GS_HOME/python/graphstorm/data/tools/preprocess_movielens.py \
    --input_path ml-100k --output_path ml-json --num_split_files 6
rm -R ml-100k


export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tests/end2end-tests/generate_test_data.py --path /$GS_HOME/python/graphstorm/data/test/data/edge_class/

# movielens node class with balanced training set
export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/construct_graph.py --name movie-lens-100k\
    --filepath movielen-data \
    --output data \
    --dist_output movielen_100k_train_val_1p_4t \
    --num_dataset_workers 10 \
    --hf_bert_model bert-base-uncased \
    --ntext_fields "movie:title" \
    --nlabel_fields "movie:genre" \
    --ntask_types "movie:classify" \
    --predict_ntype movie \
    --num_parts 1 \
    --num_trainers_per_machine 4 \
    --balance_train \
    --balance_edges \
    --generate_new_split true \
    --compute_bert_emb true \
    --device 0 \
    --remove_text_tokens true \
    --undirected

export PYTHONPATH=$GS_HOME/python/
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

# movielens node class
export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/construct_graph.py --name movie-lens-100k\
    --filepath movielen-data \
    --output data \
    --dist_output movielen_no_edata_100k_train_val_1p_4t \
    --num_dataset_workers 10 \
    --hf_bert_model bert-base-uncased \
    --ntext_fields "movie:title" \
    --nlabel_fields "movie:genre" \
    --ntask_types "movie:classify" \
    --predict_ntype movie \
    --num_parts 1 \
    --num_trainers_per_machine 4 \
    --balance_edges \
    --generate_new_split false \
    --compute_bert_emb true \
    --device 0 \
    --remove_text_tokens true \
	--undirected

# movielens edge regression
export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/construct_graph.py --name movie-lens-100k \
	--undirected \
    --filepath movielen-data \
    --output er \
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

export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/construct_graph.py --name test \
    --filepath /$GS_HOME/python/graphstorm/data/test/data/edge_class/ \
    --output data \
    --dist_output test_ec_1p_4t --num_dataset_workers 10 \
    --hf_bert_model bert-base-uncased \
    --ntext_fields "node:text" \
    --elabel_fields "node,r0,item:label" \
    --predict_etype node,r0,item \
	--num_parts 1 --balance_train \
    --balance_edges --num_trainers_per_machine 4 \
	--device 0 --split_etypes node,r0,item \
    --generate_new_split true

export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/construct_graph.py --name test \
    --filepath /$GS_HOME/python/graphstorm/data/test/data/edge_class/ \
    --output data \
    --dist_output test_ec_undirected_1p_4t --num_dataset_workers 10 \
    --hf_bert_model bert-base-uncased \
    --ntext_fields "node:text" \
    --elabel_fields "node,r0,item:label" \
    --predict_etype node,r0,item \
	--num_parts 1 --balance_train \
    --balance_edges --num_trainers_per_machine 4 \
	--device 0 --split_etypes node,r0,item \
    --generate_new_split true \
    --undirected


export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/construct_graph.py --name test \
    --filepath /$GS_HOME/python/graphstorm/data/test/data/edge_class/ \
    --output datanf \
    --dist_output test_ec_nodefeat_1p_4t --num_dataset_workers 10 \
    --hf_bert_model bert-base-uncased \
    --ntext_fields "node:text" \
    --elabel_fields "node,r0,item:label" \
    --nfeat_format hdf5\
    --predict_etype node,r0,item \
	--num_parts 1 --balance_train \
    --balance_edges --num_trainers_per_machine 4 \
	--device 0 --split_etypes node,r0,item \
    --generate_new_split true \
    --undirected


python3 /$GS_HOME/tests/end2end-tests/data_gen/gen_multi_feat_nc.py --path /data/multi-feat-nc-test/

python3 /$GS_HOME/tools/construct_graph.py --name multi-feat-nc-test \
    --filepath /data/multi-feat-nc-test/ \
    --output /data/multi-feat-nc/ \
    --dist_output test_multi_feat_nc_4t --num_dataset_workers 2 \
    --hf_bert_model bert-base-uncased \
    --nlabel_fields "ntype1:label" \
    --nfeat_format npy\
    --predict_ntypes ntype1 \
	--num_parts 1 --balance_train \
    --balance_edges --num_trainers_per_machine 4 \
	--device 0 \
    --generate_new_split true \
    --undirected

python3 /$GS_HOME/tests/end2end-tests/data_gen/gen_multi_feat_nc.py --path /data/multi-feat-same-name-nc-test/ --same-fname true

python3 /$GS_HOME/tools/construct_graph.py --name multi-feat-sn-nc-test \
    --filepath /data/multi-feat-same-name-nc-test/ \
    --output /data/multi-feat-same-name-nc/ \
    --dist_output test_multi_feat_same_name_nc_4t --num_dataset_workers 2 \
    --hf_bert_model bert-base-uncased \
    --nlabel_fields "ntype1:label" \
    --nfeat_format npy\
    --predict_ntypes ntype1 \
	--num_parts 1 --balance_train \
    --balance_edges --num_trainers_per_machine 4 \
	--device 0 \
    --generate_new_split true \
    --undirected

export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tests/end2end-tests/data_gen/gen_multilabel_nc.py --path /data/multilabel-nc-test/

python3 /$GS_HOME/tools/construct_graph.py --name multilabel-nc-test \
    --filepath /data/multilabel-nc-test/ \
    --output /data/multilabel-nc/ \
    --dist_output test_multilabel_nc_1p_4t --num_dataset_workers 2 \
    --hf_bert_model bert-base-uncased \
    --nlabel_fields "ntype1:label" \
    --nfeat_format npy\
    --predict_ntypes ntype1 \
	--num_parts 1 --balance_train \
    --balance_edges --num_trainers_per_machine 4 \
	--device 0 \
    --generate_new_split true \
    --undirected

export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tests/end2end-tests/data_gen/gen_multilabel_ec.py --path /data/multilabel-ec-test/

python3 /$GS_HOME/tools/construct_graph.py --name multilabel-ec-test \
    --filepath /data/multilabel-ec-test/ \
    --output /data/multilabel-ec/ \
    --dist_output test_multilabel_ec_1p_4t --num_dataset_workers 2 \
    --hf_bert_model bert-base-uncased \
    --elabel_fields "ntype0,r1,ntype1:label" \
    --nfeat_format npy\
    --etask_types "ntype0,r1,ntype1:classify" \
    --predict_etypes "ntype0,r1,ntype1" \
    --num_parts 1 --balance_train \
    --balance_edges --num_trainers_per_machine 4 \
	--device 0 \
    --generate_new_edge_split true \
    --undirected
