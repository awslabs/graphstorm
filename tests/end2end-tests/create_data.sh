date
GS_HOME=$(pwd)
export PYTHONPATH=$GS_HOME/python/
mkdir -p /data
cd /data
cp -R /storage/ml-100k /data

# Generate movielens dataset in gconstruct input format
python3 $GS_HOME/tests/end2end-tests/data_gen/process_movielens.py

# movielens node classification with balanced training set - used in both single and multi gpu tests
python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens.json \
	--num-processes 1 \
	--output-dir movielen_100k_train_val_1p_4t \
	--graph-name movie-lens-100k \
	--add-reverse-edges

# movielens node classification removing test mask
cp -R /data/movielen_100k_train_val_1p_4t /data/movielen_100k_train_notest_1p_4t
python3 $GS_HOME/tests/end2end-tests/data_gen/remove_test_mask.py --dataset movielen_100k_train_notest_1p_4t --remove_node_mask true

# movielens node classification inference
cp -R /data/movielen_100k_train_val_1p_4t /data/movielen_100k_infer_val_1p_4t

# movielens node classification with text features - used in multi-gpu test, 4 trainers
python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_text.json \
	--num-processes 1 \
	--output-dir movielen_100k_text_train_val_1p_4t \
	--graph-name movie-lens-100k-text \
	--add-reverse-edges

# movielens link prediction - used in both single and multi gpu tests
python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_lp.json \
	--num-processes 1 \
	--output-dir movielen_100k_lp_train_val_1p_4t \
	--graph-name movie-lens-100k \
	--add-reverse-edges

# movielens link prediction removing test mask
cp -R /data/movielen_100k_lp_train_val_1p_4t /data/movielen_100k_lp_train_no_test_1p_4t
python3 $GS_HOME/tests/end2end-tests/data_gen/remove_test_mask.py --dataset movielen_100k_lp_train_no_test_1p_4t --remove_node_mask false

# movielens link prediction with text features - used in multi-gpu test, 4 trainers
python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_text.json \
	--num-processes 1 \
	--output-dir movielen_100k_text_lp_train_val_1p_4t \
	--graph-name movie-lens-100k-text \
	--add-reverse-edges

# movielens link prediction without data split - used in single-gpu tests only
rm -Rf movielen_no_edata_100k_train_val_1p_4t
cp -R movielen_100k_lp_train_val_1p_4t movielen_no_edata_100k_train_val_1p_4t
python3 /$GS_HOME/tests/end2end-tests/data_gen/remove_mask.py --dataset movielen_no_edata_100k_train_val_1p_4t --remove_node_mask 0

# movielens edge regression - used in both single and multi gpu tests
python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_er.json \
	--num-processes 1 \
	--output-dir movielen_100k_er_1p_4t \
	--graph-name movie-lens-100k \
	--add-reverse-edges

# movielens edge regression removing test mask
cp -R /data/movielen_100k_er_1p_4t /data/movielen_100k_er_no_test_1p_4t
python3 $GS_HOME/tests/end2end-tests/data_gen/remove_test_mask.py --dataset movielen_100k_er_no_test_1p_4t --remove_node_mask false

#movielens edge regression - inference
cp -R /data/movielen_100k_er_1p_4t /data/movielen_100k_er_infer_1p_4t

# movielens edge classification - used in both single and multi-gpu tests
cp -R /data/movielen_100k_train_val_1p_4t /data/movielen_100k_ec_1p_4t

cp -R /data/movielen_100k_ec_1p_4t /data/movielen_100k_ec_no_test_1p_4t
python3 $GS_HOME/tests/end2end-tests/data_gen/remove_test_mask.py --dataset movielen_100k_ec_no_test_1p_4t --remove_node_mask false

# movielens edge classification - inference
cp -R /data/movielen_100k_ec_1p_4t /data/movielen_100k_multi_label_ec_infer

# Create data for edge classification with text features
cp -R /data/movielen_100k_text_train_val_1p_4t /data/movielen_100k_ec_1p_4t_text

# movielens edge classification without data split - used in both single and multi-gpu tests
rm -Rf /data/movielen_100k_multi_label_ec
cp -R /data/movielen_100k_ec_1p_4t /data/movielen_100k_multi_label_ec
python3 $GS_HOME/tests/end2end-tests/data_gen/gen_multilabel.py --path /data/movielen_100k_multi_label_ec --node_class false --field rate

# Create data for graph-aware fine-tuning BERT model
python3 -m graphstorm.gconstruct.construct_graph --conf-file $GS_HOME/tests/end2end-tests/test_data_config/movielens_user_feat_movie_token.json --num-processes 1 --output-dir /data/movielen_100k_lp_user_feat_movie_token_1p --graph-name ml --add-reverse-edges --num-parts 1

# For Custom GNN test 
python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_custom.json \
	--num-processes 1 \
	--output-dir movielen_100k_custom_train_val_1p_4t \
	--graph-name movie-lens-100k \
	--add-reverse-edges

# Adding old commands for Multi-GPU tests

# movielens node classifcation with balanced training set - Multi GPU
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k \
	--filepath /data \
	--target-ntype movie \
	--add-reverse-edges \
	--num-trainers-per-machine 4 \
	--output movielen_100k_train_val_1p_4t_mgpu \
	--generate-new-node-split true \
	--balance-train \
	--balance-edges \
	--num-parts 1

# node classification inference
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k \
	--filepath /data \
	--target-ntype movie \
	--add-reverse-edges \
	--num-trainers-per-machine 4 \
	--output movielen_100k_infer_val_1p_4t_mgpu \
	--no-split true \
	--balance-edges \
	--num-parts 1

# movielens node classification text features
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k-text \
	--filepath /data \
	--target-ntype movie \
	--add-reverse-edges \
	--num-trainers-per-machine 4 \
	--output movielen_100k_text_train_val_1p_4t_mgpu \
	--generate-new-node-split true \
	--balance-train \
	--balance-edges \
	--num-parts 1

# movielens link prediction
python3 /$GS_HOME/tools/partition_graph_lp.py --dataset movie-lens-100k \
	--filepath /data \
	--target-etype "user,rating,movie" \
	--num-trainers-per-machine 4 \
	--output movielen_100k_lp_train_val_1p_4t_mgpu \
	--balance-train \
	--balance-edges \
	--train-pct 0.1 \
	--val-pct 0.1 \
	--num-parts 1

# movielens link prediction text features
python3 /$GS_HOME/tools/partition_graph_lp.py --dataset movie-lens-100k-text  \
	--filepath /data \
	--target-etype "user,rating,movie" \
	--num-trainers-per-machine 4 \
	--output movielen_100k_text_lp_train_val_1p_4t_mgpu \
	--balance-train \
	--balance-edges \
	--train-pct 0.1 \
	--val-pct 0.1 \
	--num-parts 1

# movielens edge classification
export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k \
	--filepath /data \
    --elabel-field "user,rating,movie:rate" \
    --target-etype "user,rating,movie" \
    --etask-type "classification" \
	--num-trainers-per-machine 4 \
	--output movielen_100k_ec_1p_4t_mgpu \
	--balance-train \
	--balance-edges \
	--generate-new-edge-split true \
	--train-pct 0.1 \
	--val-pct 0.1 \
	--num-parts 1

rm -Rf /data/movielen_100k_multi_label_ec_mgpu
cp -R /data/movielen_100k_ec_1p_4t_mgpu /data/movielen_100k_multi_label_ec_mgpu
python3 $GS_HOME/tests/end2end-tests/data_gen/gen_multilabel.py --path /data/movielen_100k_multi_label_ec_mgpu --node_class false --field rate

# movielens edge classification text features
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k-text \
	--filepath /data \
    --elabel-field "user,rating,movie:rate" \
    --target-etype "user,rating,movie" \
    --etask-type "classification" \
	--num-trainers-per-machine 4 \
	--output movielen_100k_ec_1p_4t_text_mgpu \
	--balance-train \
	--balance-edges \
	--generate-new-edge-split true \
	--train-pct 0.1 \
	--val-pct 0.1 \
	--num-parts 1

# movielens edge classification inference
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k \
	--filepath /data \
    --elabel-field "user,rating,movie:rate" \
    --target-etype "user,rating,movie" \
    --etask-type "classification" \
	--num-trainers-per-machine 4 \
	--output movielen_100k_multi_label_ec_infer_mgpu \
	--balance-edges \
	--no-split true \
	--num-parts 1
	
date

echo 'Done'