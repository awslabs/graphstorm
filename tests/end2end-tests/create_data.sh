date
GS_HOME=$(pwd)
export PYTHONPATH=$GS_HOME/python/
mkdir -p /data
cd /data
cp -R /storage/ml-100k /data

python3 $GS_HOME/tests/end2end-tests/data_gen/process_movielens.py

# movielens node class with balanced training set
python3 -m graphstorm.gconstruct.construct_graph \
	--conf_file $GS_HOME/tests/end2end-tests/data_gen/movielens.json \
	--num_processes 1 \
	--output_dir movielen_100k_train_val_1p_4t \
	--graph_name movie-lens-100k \
	--add_reverse_edges

python3 -m graphstorm.gconstruct.construct_graph \
	--conf_file $GS_HOME/tests/end2end-tests/data_gen/movielens_text.json \
	--num_processes 1 \
	--output_dir movielen_100k_text_train_val_1p_4t \
	--graph_name movie-lens-100k-text \
	--add_reverse_edges

# movielens link prediction
python3 -m graphstorm.gconstruct.construct_graph \
	--conf_file $GS_HOME/tests/end2end-tests/data_gen/movielens_lp.json \
	--num_processes 1 \
	--output_dir movielen_100k_lp_train_val_1p_4t \
	--graph_name movie-lens-100k \
	--add_reverse_edges

# movielens link prediction with text features
#python3 -m graphstorm.gconstruct.construct_graph \
#	--conf_file $GS_HOME/tests/end2end-tests/data_gen/movielens_lp_text.json \
#	--num_processes 1 \
#	--output_dir movielen_100k_text_lp_train_val_1p_4t \
#	--graph_name movie-lens-100k-text \
#	--add_reverse_edges

# movielens link prediction without data split.
rm -Rf movielen_no_edata_100k_train_val_1p_4t
cp -R movielen_100k_lp_train_val_1p_4t movielen_no_edata_100k_train_val_1p_4t
python3 /$GS_HOME/tests/end2end-tests/data_gen/remove_mask.py --dataset movielen_no_edata_100k_train_val_1p_4t --remove_node_mask 0

# movielens edge regression
python3 -m graphstorm.gconstruct.construct_graph \
	--conf_file $GS_HOME/tests/end2end-tests/data_gen/movielens_er.json \
	--num_processes 1 \
	--output_dir movielen_100k_er_1p_4t \
	--graph_name movie-lens-100k \
	--add_reverse_edges

# dummy data Edge Classification
python3 -m graphstorm.gconstruct.construct_graph \
	--conf_file $GS_HOME/tests/end2end-tests/data_gen/movielens.json \
	--num_processes 1 \
	--output_dir movielen_100k_ec_1p_4t \
	--graph_name movie-lens-100k \
	--add_reverse_edges

rm -Rf /data/movielen_100k_multi_label_ec
cp -R /data/movielen_100k_ec_1p_4t /data/movielen_100k_multi_label_ec
python3 $GS_HOME/tests/end2end-tests/data_gen/gen_multilabel.py --path /data/movielen_100k_multi_label_ec --node_class false --field rate

date
