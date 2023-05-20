# Amazon SageMaker Support
This submodule provides support to run GraphStorm training and inference on Amazon SageMaker.
To build a SageMaker compatible docker image, please refer to [Build GraphStorm SageMaker docker image] (https://github.com/awslabs/graphstorm/docker/sagemaker).

## Launch SageMaker tasks
Use scripts under graphstorm.sagemaker.launch to launch SageMaker tasks.
Please make sure you already setup your SageMaker environments.
Please refer to [Amazon SageMaker service](https://aws.amazon.com/pm/sagemaker) for how to get access to Amazon SageMaker.

### Launch GraphStorm training/inference using Amazon SageMaker service

#### Launch train task using built-in training script

##### Preparing training data and training task config.
```
cd ~/
git clone https://github.com/awslabs/graphstorm.git
GS_HOME=~/graphstorm/
python3 $GS_HOME/tools/partition_graph.py --dataset ogbn-arxiv \
                                          --filepath /tmp/ogbn-arxiv-nc/ \
                                          --num_parts 2 \
                                          --output /tmp/ogbn_arxiv_nc_2p
```

You need to upload /tmp/ogbn_arxiv_nc_2p into S3. You also need to upload the yaml config file into S3.
You can find the example yaml file in https://github.com/awslabs/graphstorm/blob/main/training_scripts/gsgnn_np/arxiv_nc.yaml.

##### Launch train task
```
python3 -m graphstorm.sagemaker.launch.launch_train \
        --image-url <AMAZON_ECR_IMAGE_PATH> \
        --region us-east-1 \
        --entry-point run/sagemaker_train.py \
        --role <ARN_ROLE> \
        --graph-data-s3 s3://PATH_TO/ogbn_arxiv_nc_2p/ \
        --yaml-s3 <S3_PATH_TO_TRAINING_CONFIG> \
        --model-artifact-s3 <S3_PATH_TO_SAVE_TRAINED_MODEL> \
        --graph-name ogbn-arxiv \
        --task-type "node_classification" \
        --num-layers 1 \
        --hidden-size 128 \
        --backend gloo \
        --batch-size 128 \
        --node-feat-name node:feat
```
The trained model artifact will be stored in the S3 address provided through `--model-artifact-s3`.
Please note `save_embed_path` and `save_prediction_path` must be disabled, i.e., set to 'None' when using SageMaker.
They only work with shared file system while SageMaker solution does not support using shared file system now.


### Launch inference task using built-in inference script
Inference task can use the same graph as training task. You can also run inference on a new graph.
In this example, we will use the same graph.

Launch inference task
```
python3 -m graphstorm.sagemaker.launch.launch_infer \
        --image-url <AMAZON_ECR_IMAGE_PATH> \
        --region us-east-1 \
        --entry-point run/sagemaker_infer.py \
        --role <ARN_ROLE> \
        --graph-data-s3 s3://PATH_TO/ogbn_arxiv_nc_2p/ \
        --yaml-s3 <S3_PATH_TO_TRAINING_CONFIG> \
        --model-artifact-s3 <S3_PATH_TO_SAVED_MODEL> \
        --output-emb-s3 <S3_PATH_TO_SAVE_GENERATED_NODE_EMBEDDING> \
        --output-prediction-s3 <S3_PATH_TO_SAVE_PREDICTION_RESULTS> \
        --graph-name ogbn-arxiv \
        --task-type "node_classification" \
        --num-layers 1 \
        --hidden-size 128 \
        --backend gloo \
        --batch-size 128 \
        --node-feat-name node:feat
```

### Test GraphStorm SageMaker runs locally with Docker compose
This section describes how to launch Docker compose jobs that emulate a SageMaker
training execution environment that can be used to test GraphStorm model training
and inference using SageMaker.


#### TLDR

1. Install Docker and docker compose: https://docs.docker.com/compose/install/linux/
2. Clone graph-storm.
3. Build the SageMaker graph-storm Docker image.
4. Generate a docker compose file:
```
python3 -m graphstorm.sagemaker.local.generate_sagemaker_docker_compose --num-instances $NUM_INSTANCES --input-data $DATASET_S3_PATH --output-data-s3 "s3://${OUTPUT_BUCKET}/test/${DATASET_NAME}${PATH_SUFFIX}/${NUM_INSTANCES}x-${INSTANCE_TYPE}/" --region $REGION
```

Launch the job using docker compose:
```
docker compose -f "docker-compose-${GRAPH_NAME}-${NUM_INSTANCES}workers-${NUM_PARTITIONS}parts.yml" up
```

#### Getting started
If you’ve never worked with Docker compose before the official description provides a great intro:

> Compose is a tool for defining and running multi-container Docker applications.
With Compose, you use a YAML file to configure your application’s services.
Then, with a single command, you create and start all the services from your configuration.

We will use this capability to launch multiple worker instances locally, that will
be configured to “look like” a SageMaker training instance and communicate over a
virtual network created by Docker compose. This way our test environment will be
as close to a real SageMaker distributed job as we can get, without needing to
launch SageMaker jobs, or launch and configure multiple EC2 instances when
developing features.

#### Prerequisite 1: Launch or re-use a GPU EC2 instance
As we will be running multiple heavy containers is one machine we recommend using
a capable Linux-based machine equiped with GPUs. We recommend at least 32GB of RAM.


#### Prerequisite 2: Install Docker and docker compose

You can follow the official Docker guides for [installation of the Docker engine](https://docs.docker.com/engine/install/).

Next you need to install the `Docker compose` plugin that will allow us to spin up
multiple Docker containers. Instructions for that are [here](https://docs.docker.com/compose/install/linux/).

#### Building the SageMaker GraphStorm docker image
Following [Build GraphStorm SageMaker docker image] (https://github.com/awslabs/graphstorm/docker/sagemaker) to build your own SageMaker GraphStorm docker image.

#### Creating the Docker compose file and run
A Docker compose file is a YAML file that tells Docker which containers to spin up and how to configure them.
To launch the services with a docker compose file, we can use `docker compose -f docker-compose.yaml up`.
This will launch the container and execute its entry point.

To emulate a SageMaker distributed execution environment based on the image (suppose the docker image is named graphstorm-sagemaker-dev:v1) you built previously you would need a Docker compose file that looks like this:
```
version: '3.7'

networks:
  gfs:
    name: gfs-network

services:
  algo-1:
    image: graphstorm-sagemaker-dev:v1
    container_name: algo-1
    hostname: algo-1
    networks:
      - gfs
    command: 'xxx'
    environment:
      SM_TRAINING_ENV: '{"hosts": ["algo-1", "algo-2", "algo-3", "algo-4"], "current_host": "algo-1"}'
      WORLD_SIZE: 4
      MASTER_ADDR: 'algo-1'
      AWS_REGION: 'us-west-2'
    ports:
      - 22
    working_dir: '/opt/ml/code/'

  algo-2:
      [...]

```
Some explanation on the above elements (see the [official docs](https://docs.docker.com/compose/compose-file/) for more details):

* `image`: Determines which image you will use for the container launched.
* `environment`: Determines the environment variables that will be set for the container once it launches.
* `command`: Determines the entrypoint, i.e. the command that will be executed once the container launches.

To help you generate yaml file automatically, we provide a Python script that
builds the docker compose file for your, `generate_sagemaker_docker_compose.py`.
Note that the script uses the [PyYAML](https://pypi.org/project/PyYAML/) library.

This file has 4 required arguments that determine the Docker compose file that will be generated:

* `--num-instances`: The number of instances we want to launch.
This will determine the number of `algo-x` `services` entries our compose file ends up with.
* `--aws-access-key-id`: The AWS access key ID for accessing S3 data within docker
* `--aws-secret-access-key`: The AWS secret access key for accessing S3 data within docker.
* `--aws-session-token`: The AWS session toekn used for accessing S3 data within docker.

The rest of the arguments are passed on to `sagemaker_train.py`
* `--task-type`: Training task type.
* `--graph-data-s3`: S3 location of input training graph.
* `--graph-name`: Name of the input training graph.
* `--train-yaml-s3`: S3 location of training yaml file.
* `--custom-script`: Custom training script provided by a customer to run customer training logic. This should be a path to the python script within the docker image.

If you want to pass other arguements to `sagemaker_train.py`, you can simply append those arguments after the above arguments.

The above will create a Docker compose file named `docker-compose-${task-type}-${num-instances}-train.yml`, which we can then use to launch the job with (for example):

```bash
docker compose -f docker-compose-node-classification-4-train.yml up
```

Running the above command will launch 4 instances of the image, configured with
the command and env vars that emulate a SageMaker execution environment and run
the `sagemaker_train.py` code. Note that the containers actually
interact with S3 so you would require valid AWS credentials to run.

#### Dcoker compose for inference
