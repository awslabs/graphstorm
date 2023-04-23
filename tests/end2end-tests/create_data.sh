date
GS_HOME=$(pwd)
export PYTHONPATH=$GS_HOME/python/
mkdir -p /data
cd /data
cp -R /storage/ml-100k /data

python3 $GS_HOME/tests/end2end-tests/data_process/process_movielens.py

# movielens node class with balanced training set
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k \
	--filepath /data \
	--target_ntype movie \
	--undirected \
	--num_trainers_per_machine 4 \
	--output movielen_100k_train_val_1p_4t \
	--generate_new_node_split true \
	--balance_train \
	--balance_edges \
	--num_parts 1

python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k-text \
	--filepath /data \
	--target_ntype movie \
	--undirected \
	--num_trainers_per_machine 4 \
	--output movielen_100k_text_train_val_1p_4t \
	--generate_new_node_split true \
	--balance_train \
	--balance_edges \
	--num_parts 1

# movielens link prediction
python3 /$GS_HOME/tools/partition_graph_lp.py --dataset movie-lens-100k \
	--filepath /data \
	--target_etype "user,rating,movie" \
	--num_trainers_per_machine 4 \
	--output movielen_100k_lp_train_val_1p_4t \
	--balance_train \
	--balance_edges \
	--train_pct 0.1 \
	--val_pct 0.1 \
	--num_parts 1

# movielens link prediction with text features
python3 /$GS_HOME/tools/partition_graph_lp.py --dataset movie-lens-100k-text  \
	--filepath /data \
	--target_etype "user,rating,movie" \
	--num_trainers_per_machine 4 \
	--output movielen_100k_text_lp_train_val_1p_4t \
	--balance_train \
	--balance_edges \
	--train_pct 0.1 \
	--val_pct 0.1 \
	--num_parts 1

# movielens node class
rm -Rf movielen_no_edata_100k_train_val_1p_4t
cp -R movielen_100k_lp_train_val_1p_4t movielen_no_edata_100k_train_val_1p_4t
python3 /$GS_HOME/tests/end2end-tests/data_gen/remove_mask.py --dataset movielen_no_edata_100k_train_val_1p_4t --remove_node_mask 0

# movielens edge regression
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k \
	--filepath /data \
    --elabel_field "user,rating,movie:rate" \
    --target_etype "user,rating,movie" \
    --etask_type "regression" \
	--num_trainers_per_machine 4 \
	--output movielen_100k_er_1p_4t \
	--balance_train \
	--balance_edges \
	--generate_new_edge_split true \
	--train_pct 0.1 \
	--val_pct 0.1 \
	--num_parts 1

# dummy data Edge Classification
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k \
	--filepath /data \
    --elabel_field "user,rating,movie:rate" \
    --target_etype "user,rating,movie" \
    --etask_type "classification" \
	--num_trainers_per_machine 4 \
	--output movielen_100k_ec_1p_4t \
	--balance_train \
	--balance_edges \
	--generate_new_edge_split true \
	--train_pct 0.1 \
	--val_pct 0.1 \
	--num_parts 1

# Create data for edge classification with text features
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k-text \
	--filepath /data \
    --elabel_field "user,rating,movie:rate" \
    --target_etype "user,rating,movie" \
    --etask_type "classification" \
	--num_trainers_per_machine 4 \
	--output movielen_100k_ec_1p_4t_text \
	--balance_train \
	--balance_edges \
	--generate_new_edge_split true \
	--train_pct 0.1 \
	--val_pct 0.1 \
	--num_parts 1

rm -Rf /data/movielen_100k_multi_label_ec
cp -R /data/movielen_100k_ec_1p_4t /data/movielen_100k_multi_label_ec
python3 $GS_HOME/tests/end2end-tests/data_gen/gen_multilabel.py --path /data/movielen_100k_multi_label_ec --node_class false --field rate

date
