# GraphStorm SageMaker support
Here we provide SageMaker integration for GraphStorm. We assume all data are stored in S3

## How to build the docker image
```
git clone git@ssh.gitlab.aws.dev:agml/graph-storm.git
cp graphstorm/sagemaker/docker/changehostname.c ./
cp graphstorm/sagemaker/docker/start_with_right_hostname.sh ./

docker build -f graphstorm/sagemaker/docker/Dockerfile . -t <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/graphstorm_train:sagemaker
```

## How to launch SageMaker task

### Link prediction
```
python3 graphstorm/sagemaker/launch_train.py  --version-tag sagemaker --training-ecr-repository graphstorm_train --account-id <ACCOUNT_ID> --region us-east-1 --role <ARN:ROLE> --graph-name ogbn-arxiv --graph-data-s3 <S3_PATH_TO_GRAPH_DATA> --task-type "link_prediction" --train-yaml-s3 <S3_PATH_TO_TRAINING_YAML_CONFIG> --train-yaml-name arxiv_lp_hf.yaml --n-layers 1 --n-hidden 128
```
