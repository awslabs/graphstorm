# GraphStorm SageMaker support
Here we provide SageMaker integration for GraphStorm. We assume all data are stored in S3

## How to build the docker image
```
git clone https://github.com/awslabs/graphstorm

docker build -f graphstorm/sagemaker/docker/Dockerfile . -t <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/graphstorm_train:sagemaker
```

## How to launch SageMaker task
See sagemaker/examples for more detials. We provide example scripts for five different tasks include node classification, node regression, edge classification, edge regression and link prediction.

## How to launch distribute ParMetis for GraphStorm
We provide an script to launch distributed ParMetis pipeline. The input should be an S3 location storing data following the format as [TBD]. The output will be partitioned DGL graph.
```
python3 graphstorm/sagemaker/launch_parmetis.py --graph-name <Graph_Name> --graph-data-s3 <S3_PATH_TO_INPUT_GRAPH > --num-parts <NUMBER_OF_PARTITIONS> --output-data-s3 <S3_PATH_TO_STORE_OUTPUT_GRAPH> --image-url <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/graphstorm_train:sagemaker --region <REGION> --role <ROLE>
```
