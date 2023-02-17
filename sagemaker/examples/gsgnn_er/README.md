# Example to launch link prediction tasks using SageMaker

## Data preparation
Create partitioned data using movielens dataset.
```
$ GS_HOME=/graph-storm
$ export PYTHONPATH=$GS_HOME/python/
$ wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
$ unzip ml-100k.zip
$ rm ml-100k.zip
$ python3 $GS_HOME/tools/gen_movielens_dataset.py --filepath ./ --savepath movielen-data --max_sequence_length 64 --retain_original_features True --user_age_as_label true --edge_pct 0.9

$ rm -R ml-100k
$ python3 -u $GS_HOME/tools/partition_graph_lp.py \
    --dataset movie-lens-100k \
    --filepath movielen-data \
    --num_parts 2 \
    --num_trainers_per_machine 4 \
    --output movielen_100k_er_2p_4t \
    --elabel_fields "user,rating,movie:rate" \
    --predict_etypes "user,rating,movie" \
    --etask_types "user,rating,movie:regression" \
    --balance_train \
    --balance_edges
```

Upload partitioned data into S3
```
aws s3 cp --recursive movielen_100k_er_2p_4t <S3_PATH_TO_GRAPH_DATA>
```

Upload training config yaml file into s3
```
aws s3 cp --recursive $GS_HOME/training_scripts/gsgnn_er/ml_er.yaml <S3_PATH_TO_TRAIN_CONFIG>/ml_er.yaml
```

## Run training
Modify train.sh by substituting ACCOUNT_ID, IAM_ROLE, S3_PATH_TO_GRAPH_DATA, S3_PATH_TO_STORE_SAVED_MODEL and S3_PATH_TO_TRAIN_CONFIG with your own data.

```
sh train.sh
```

## Run inference
Find the path to the saved SageMaker model artifact under S3_PATH_TO_STORE_SAVED_MODEL. The S3_PATH_TO_MODEL_TO_BE_LOAD and MODEL_CHECKPOINT_NAME should be derived from S3_PATH_TO_STORE_SAVED_MODEL.

Modify infer.sh by substituting ACCOUNT_ID, IAM_ROLE, S3_PATH_TO_GRAPH_DATA, S3_PATH_TO_MODEL_TO_BE_LOAD, MODEL_CHECKPOINT_NAME, S3_PATH_TO_INFER_CONFIG and S3_PATH_TO_UPLOAD_EMB with your own data.

```
sh infer.sh
```
