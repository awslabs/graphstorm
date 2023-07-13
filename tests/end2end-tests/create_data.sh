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

# movielens node class with balanced training set
export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k \
	--filepath /data \
	--target-ntype movie \
	--add-reverse-edges \
	--num-trainers-per-machine 4 \
	--output movielen_100k_train_val_1p_4t \
	--generate-new-node-split true \
	--balance-train \
	--balance-edges \
	--num-parts 1

python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k \
	--filepath /data \
	--target-ntype movie \
	--add-reverse-edges \
	--num-trainers-per-machine 4 \
	--output movielen_100k_infer_val_1p_4t \
	--no-split true \
	--balance-edges \
	--num-parts 1

export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k-text \
	--filepath /data \
	--target-ntype movie \
	--add-reverse-edges \
	--num-trainers-per-machine 4 \
	--output movielen_100k_text_train_val_1p_4t \
	--generate-new-node-split true \
	--balance-train \
	--balance-edges \
	--num-parts 1

# movielens link prediction
export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/partition_graph_lp.py --dataset movie-lens-100k \
	--filepath /data \
	--target-etype "user,rating,movie" \
	--num-trainers-per-machine 4 \
	--output movielen_100k_lp_train_val_1p_4t \
	--balance-train \
	--balance-edges \
	--train-pct 0.1 \
	--val-pct 0.1 \
	--num-parts 1

# movielens link prediction with text features
export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/partition_graph_lp.py --dataset movie-lens-100k-text  \
	--filepath /data \
	--target-etype "user,rating,movie" \
	--num-trainers-per-machine 4 \
	--output movielen_100k_text_lp_train_val_1p_4t \
	--balance-train \
	--balance-edges \
	--train-pct 0.1 \
	--val-pct 0.1 \
	--num-parts 1

# movielens node class
export PYTHONPATH=$GS_HOME/python/
rm -Rf movielen_no_edata_100k_train_val_1p_4t
cp -R movielen_100k_lp_train_val_1p_4t movielen_no_edata_100k_train_val_1p_4t
python3 /$GS_HOME/tests/end2end-tests/data_gen/remove_mask.py --dataset movielen_no_edata_100k_train_val_1p_4t --remove_node_mask 0

# movielens edge regression
export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k \
	--filepath /data \
    --elabel-field "user,rating,movie:rate" \
    --target-etype "user,rating,movie" \
    --etask-type "regression" \
	--num-trainers-per-machine 4 \
	--output movielen_100k_er_1p_4t \
	--balance-train \
	--balance-edges \
	--generate-new-edge-split true \
	--train-pct 0.1 \
	--val-pct 0.1 \
	--num-parts 1

python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k \
	--filepath /data \
    --elabel-field "user,rating,movie:rate" \
    --target-etype "user,rating,movie" \
    --etask-type "regression" \
	--num-trainers-per-machine 4 \
	--output movielen_100k_er_infer_1p_4t \
	--balance-edges \
	--no-split true \
	--num-parts 1

# dummy data Edge Classification
export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k \
	--filepath /data \
    --elabel-field "user,rating,movie:rate" \
    --target-etype "user,rating,movie" \
    --etask-type "classification" \
	--num-trainers-per-machine 4 \
	--output movielen_100k_ec_1p_4t \
	--balance-train \
	--balance-edges \
	--generate-new-edge-split true \
	--train-pct 0.1 \
	--val-pct 0.1 \
	--num-parts 1

export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k \
	--filepath /data \
    --elabel-field "user,rating,movie:rate" \
    --target-etype "user,rating,movie" \
    --etask-type "classification" \
	--num-trainers-per-machine 4 \
	--output movielen_100k_multi_label_ec_infer \
	--balance-edges \
	--no-split true \
	--num-parts 1

# Create data for edge classification with text features
export PYTHONPATH=$GS_HOME/python/
python3 /$GS_HOME/tools/partition_graph.py --dataset movie-lens-100k-text \
	--filepath /data \
    --elabel-field "user,rating,movie:rate" \
    --target-etype "user,rating,movie" \
    --etask-type "classification" \
	--num-trainers-per-machine 4 \
	--output movielen_100k_ec_1p_4t_text \
	--balance-train \
	--balance-edges \
	--generate-new-edge-split true \
	--train-pct 0.1 \
	--val-pct 0.1 \
	--num-parts 1

rm -Rf /data/movielen_100k_multi_label_ec
cp -R /data/movielen_100k_ec_1p_4t /data/movielen_100k_multi_label_ec
python3 $GS_HOME/tests/end2end-tests/data_gen/gen_multilabel.py --path /data/movielen_100k_multi_label_ec --node_class false --field rate

date
