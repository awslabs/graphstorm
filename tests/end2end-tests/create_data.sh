date
GS_HOME=$(pwd)
mkdir -p /data
cd /data
cp -R /storage/ml-100k /data

python3 /$GS_HOME/python/graphstorm/data/tools/preprocess_movielens.py \
    --input_path ml-100k --output_path movielen-data
rm -fr ml-json

python3 /$GS_HOME/python/graphstorm/data/tools/preprocess_movielens.py \
    --input_path ml-100k --output_path ml-json --num_split_files 6


export PYTHONPATH=$GS_HOME/python/
mkdir -p /data/edge_class
python3 /$GS_HOME/tests/end2end-tests/generate_test_data.py --path /data/edge_class/

# movielens node class with balanced training set
export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k \
	--filepath /data \
	--predict_ntypes movie \
	--undirected \
	--num_trainers_per_machine 4 \
	--output movielen_100k_train_val_1p_4t \
	--generate_new_split true \
	--balance_train \
	--balance_edges \
	--num_parts 1

export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/partition_graph_lp.py --dataset movie-lens-100k \
	--filepath /data \
	--predict_etypes "user,rating,movie" \
	--num_trainers_per_machine 4 \
	--output movielen_100k_lp_train_val_1p_4t \
	--balance_train \
	--balance_edges \
	--edge_pct 0.8 \
	--num_parts 1

# movielens node class
export PYTHONPATH=$GS_HOME/python/
cp -R movielen_100k_lp_train_val_1p_4t movielen_no_edata_100k_train_val_1p_4t
python3 /$GS_HOME/tests/end2end-tests/data_gen/remove_mask.py --dataset movielen_no_edata_100k_train_val_1p_4t --remove_node_mask 0

# movielens edge regression
export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/partition_graph_lp.py --dataset movie-lens-100k \
	--filepath /data \
    --elabel_fields "user,rating,movie:rate" \
    --predict_etypes "user,rating,movie" \
    --etask_types "user,rating,movie:regression" \
	--num_trainers_per_machine 4 \
	--output movielen_100k_er_1p_4t \
	--balance_train \
	--balance_edges \
	--edge_pct 0.2 \
	--num_parts 1

# dummy data Edge Classification
export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/partition_graph_lp.py --dataset movie-lens-100k \
	--filepath /data \
    --elabel_fields "user,rating,movie:rate" \
    --predict_etypes "user,rating,movie" \
    --etask_types "user,rating,movie:classify" \
	--num_trainers_per_machine 4 \
	--output movielen_100k_ec_1p_4t \
	--balance_train \
	--balance_edges \
	--edge_pct 0.2 \
	--num_parts 1

cp -R /data/movielen_100k_ec_1p_4t /data/movielen_100k_multi_label_ec
python3 $GS_HOME/tests/end2end-tests/data_gen/gen_multilabel.py --path /data/movielen_100k_multi_label_ec --node_class false --field rate

date
