#!/bin/sh
python3 graphstorm/sagemaker/launch_infer.py --image-url GRAPHSTORM_IMAGE_URL --region us-east-1 --role IAM_ROLE --graph-name movie-lens-100k --graph-data-s3 S3_PATH_TO_GRAPH_DATA --task-type "edge_regression" --model-artifact-s3 S3_PATH_TO_MODEL_TO_BE_LOAD --model-sub-path MODEL_CHECKPOINT_NAME --infer-yaml-s3 S3_PATH_TO_INFER_CONFIG --infer-yaml-name ml_er.yaml --emb-s3-path S3_PATH_TO_UPLOAD_EMB --n-layers 2 --n-hidden 128 --backend gloo --batch-size 128 --fanout 10,5 --feat-name user:feat movie:feat
