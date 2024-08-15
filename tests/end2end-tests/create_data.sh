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

python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_small_val.json \
	--num-processes 1 \
	--output-dir movielen_100k_train_small_val_1p_4t \
	--graph-name movie-lens-100k \
	--add-reverse-edges

# movielens node classification removing test mask
rm -Rf /data/movielen_100k_train_notest_1p_4t
cp -R /data/movielen_100k_train_val_1p_4t /data/movielen_100k_train_notest_1p_4t
python3 $GS_HOME/tests/end2end-tests/data_gen/remove_test_mask.py --dataset movielen_100k_train_notest_1p_4t --remove_node_mask true

# movielens node classification inference
rm -Rf /data/movielen_100k_infer_val_1p_4t
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
cp -R /data/ml-100k/raw_id_mappings/ movielen_100k_lp_train_val_1p_4t/

# movielens link prediction - hard negative and fixed negative for inference
rm -Rf /data/movielen_100k_lp_train_val_hard_neg_1p_4t
python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_lp_hard.json \
	--num-processes 1 \
	--output-dir movielen_100k_lp_train_val_hard_neg_1p_4t \
	--graph-name movie-lens-100k \
	--add-reverse-edges

# movielens link prediction removing test mask
rm -Rf /data/movielen_100k_lp_train_no_test_1p_4t
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

# movielens link prediction with two edge types
python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_2etype_lp.json \
	--num-processes 1 \
	--output-dir movielen_100k_lp_2etype_train_val_1p_4t \
	--graph-name movie-lens-100k \
	--add-reverse-edges

# movielens edge regression - used in both single and multi gpu tests
python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_er.json \
	--num-processes 1 \
	--output-dir movielen_100k_er_1p_4t \
	--graph-name movie-lens-100k \
	--add-reverse-edges

# movielens edge regression removing test mask
rm -Rf /data/movielen_100k_er_no_test_1p_4t
cp -R /data/movielen_100k_er_1p_4t /data/movielen_100k_er_no_test_1p_4t
python3 $GS_HOME/tests/end2end-tests/data_gen/remove_test_mask.py --dataset movielen_100k_er_no_test_1p_4t --remove_node_mask false

#movielens edge regression - inference
rm -Rf /data/movielen_100k_er_infer_1p_4t
cp -R /data/movielen_100k_er_1p_4t /data/movielen_100k_er_infer_1p_4t

# movielens edge classification - used in both single and multi-gpu tests
rm -Rf /data/movielen_100k_ec_1p_4t
cp -R /data/movielen_100k_train_val_1p_4t /data/movielen_100k_ec_1p_4t

rm -Rf /data/movielen_100k_ec_no_test_1p_4t
cp -R /data/movielen_100k_ec_1p_4t /data/movielen_100k_ec_no_test_1p_4t
python3 $GS_HOME/tests/end2end-tests/data_gen/remove_test_mask.py --dataset movielen_100k_ec_no_test_1p_4t --remove_node_mask false

# movielens edge classification - inference
rm -Rf /data/movielen_100k_multi_label_ec_infer
cp -R /data/movielen_100k_ec_1p_4t /data/movielen_100k_multi_label_ec_infer

# Create data for edge classification with text features
rm -Rf /data/movielen_100k_ec_1p_4t_text
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

# For tests using lm-encoder - node classification and edge classification
python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_lm_encoder.json \
	--num-processes 1 \
	--output-dir movielen_100k_lm_encoder_train_val_1p_4t \
	--graph-name movie-lens-100k-text \
	--add-reverse-edges

# For tests using lm-encoder on movies - node classification and edge classification
python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_movie_lm_encoder.json \
	--num-processes 1 \
	--output-dir movielen_100k_movie_lm_encoder_train_val_1p_4t \
	--graph-name movie-lens-100k-text \
	--add-reverse-edges

# roberta as the LM:
python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_roberta_encoder.json \
	--num-processes 1 \
	--output-dir movielen_100k_roberta_encoder_train_val_1p_4t \
	--graph-name movie-lens-100k-text \
	--add-reverse-edges

# For tests using lm-encoder - link prediction
python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_lm_encoder_lp.json \
	--num-processes 1 \
	--output-dir movielen_100k_lm_encoder_lp_train_val_1p_4t \
	--graph-name movie-lens-100k-text \
	--add-reverse-edges

# movielens with labels on both user and movie nodes
python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_multi_target_ntypes.json \
	--num-processes 1 \
	--output-dir movielen_100k_multi_target_ntypes_train_val_1p_4t \
	--graph-name movie-lens-100k \
	--add-reverse-edges

python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_multi_task.json \
	--num-processes 1 \
	--output-dir movielen_100k_multi_task_train_val_1p_4t \
	--graph-name movie-lens-100k \
	--add-reverse-edges

date

echo 'Done'
