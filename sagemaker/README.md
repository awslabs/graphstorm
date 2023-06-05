# Amazon SageMaker Support
This submodule provides support to run GraphStorm graph construction, training and inference on Amazon SageMaker.
To build a SageMaker compatible docker image, please refer to [Build GraphStorm SageMaker docker image] (https://github.com/awslabs/graphstorm/docker/sagemaker).


## Launch SageMaker tasks
Use scripts under graphstorm.sagemaker.launch to launch SageMaker tasks.
Please make sure you already setup your SageMaker environment.
Please refer to [Amazon SageMaker service](https://aws.amazon.com/pm/sagemaker) for how to get access to Amazon SageMaker.

### Launch GraphStorm graph construction using Amazon SageMaker service

#### Preparing the example dataset
We use the built-in acm dataset as an example.
You can generate the raw acm dataset in parquet format using the following instructions
(See [Use Your Own Data Tutorial](https://github.com/awslabs/graphstorm/wiki/tutorials-own-data#use-own-data) for more details):
```
cd ~/
git clone https://github.com/awslabs/graphstorm.git
GS_HOME=~/graphstorm/
cd $GS_HOME/examples
python3 acm_data.py --output-path acm_raw
```
The raw graph input data will be stored at ~/graphstorm/examples/acm_raw. The input configuration JSON is also generated and stored in the same path at  ~/graphstorm/examples/acm_raw/config.json.

#### launch graph processing task
Before launching the task, you need to upload the raw acm dataset (i.e., /tmp/acm_raw) into S3.
```
aws s3 cp --recursive  ~/graphstorm/examples/acm_raw s3://PATH_TO/acm/acm_raw/
```

Then, you can use the following command to launch a SageMaker graph construction task.
```
python3 launch/launch_gconstruct.py \
        --image-url <AMAZON_ECR_IMAGE_URI> \
        --region us-east-1 \
        --entry-point run/train_entry.py \
        --role <ROLE_ARN> \
        --input-graph-s3 s3://PATH_TO/acm/ \
        --output-graph-s3 s3://PATH_TO/acm_output/ \
        --volume-size-in-gb 10 \
        --graph-name acm \
        --graph-config-file acm_raw/config.json
```
The processed graph data is stored at s3://PATH_TO/acm_output/.

Note: The `--input-graph-s3` path will be mapped into `/opt/ml/processing/input` and used as the command working directory when launching the graph construction command.
The graph configuration file should be stored with the input graph data.
The argument `--graph-config-file` should be a relative path to `--input-graph-s3`.

### Launch GraphStorm training/inference using Amazon SageMaker service

#### Launch train task using built-in training script

##### Preparing training data and training task config.
We use the built-in ogbn-arxiv dataset as an example.
First you need to partition the graph dataset by following the instructions:
```
cd ~/
git clone https://github.com/awslabs/graphstorm.git
GS_HOME=~/graphstorm/
python3 $GS_HOME/tools/partition_graph.py --dataset ogbn-arxiv \
                                          --filepath /tmp/ogbn-arxiv-nc/ \
                                          --num_parts 2 \
                                          --output /tmp/ogbn_arxiv_nc_2p
```
The partitioned graph will be stored at /tmp/ogbn_arxiv_nc_2p.

##### Launch train task
Before launching the task, you need to upload the partitioned graph (i.e., /tmp/ogbn_arxiv_nc_2p) into S3.
You also need to upload the yaml config file into S3.
You can find the example yaml file in https://github.com/awslabs/graphstorm/blob/main/training_scripts/gsgnn_np/arxiv_nc.yaml.
```
aws s3 cp --recursive /tmp/ogbn_arxiv_nc_2p s3://PATH_TO/ogbn_arxiv_nc_2p/
aws s3 cp PATH_TO/arxiv_nc.yaml s3://PATH_TO_TRAINING_CONFIG/arxiv_nc.yaml
```

Then, you can use the following command to launch a SageMaker training task.
```
cd $GS_HOME/sagemaker/
python3 launch/launch_train.py \
        --image-url <AMAZON_ECR_IMAGE_URI> \
        --region us-east-1 \
        --entry-point run/train_entry.py \
        --role <ROLE_ARN> \
        --graph-data-s3 s3://PATH_TO/ogbn_arxiv_nc_2p/ \
        --yaml-s3 s3://PATH_TO_TRAINING_CONFIG/arxiv_nc.yaml \
        --model-artifact-s3 s3://PATH_TO_SAVE_TRAINED_MODEL/ \
        --graph-name ogbn-arxiv \
        --task-type "node_classification" \
        --num-layers 1 \
        --hidden-size 128 \
        --backend gloo \
        --batch-size 128 \
        --node-feat-name node:feat
```
The trained model artifact will be stored in the S3 address provided through `--model-artifact-s3`.
You can use following command to check the model artifacts:
```
aws s3 ls s3://PATH_TO_SAVE_TRAINED_MODEL/
```

Please note `save_embed_path` and `save_prediction_path` must be disabled, i.e., set to 'None' when using SageMaker.
They only work with shared file system while SageMaker solution does not support using shared file system now.


### Launch inference task using built-in inference script
Inference task can use the same graph as training task. You can also run inference on a new graph.
In this example, we will use the same graph.

you can use the following command to launch a SageMaker offline inference task.
```
cd $GS_HOME/sagemaker/
python3 launch/launch_infer \
        --image-url <AMAZON_ECR_IMAGE_URI> \
        --region us-east-1 \
        --entry-point run/infer_entry.py \
        --role <ROLE_ARN> \
        --graph-data-s3 s3://PATH_TO/ogbn_arxiv_nc_2p/ \
        --yaml-s3 s3://PATH_TO_TRAINING_CONFIG/arxiv_nc.yaml \
        --model-artifact-s3  s3://PATH_TO_SAVED_MODEL/ \
        --output-emb-s3 s3://PATH_TO_SAVE_GENERATED_NODE_EMBEDDING/ \
        --output-prediction-s3 s3://PATH_TO_SAVE_PREDICTION_RESULTS \
        --graph-name ogbn-arxiv \
        --task-type "node_classification" \
        --num-layers 1 \
        --hidden-size 128 \
        --backend gloo \
        --batch-size 128 \
        --node-feat-name node:feat
```
The generated node embeddings will be uploaded into s3://PATH_TO_SAVE_GENERATED_NODE_EMBEDDING/.
The prediction results of node classification/regression or edge classification/regression tasks will be uploaded into s3://PATH_TO_SAVE_PREDICTION_RESULTS/
You can use following command to check the corresponding outputs:
```
aws s3 ls s3://PATH_TO_SAVE_GENERATED_NODE_EMBEDDING/
aws s3 ls s3://PATH_TO_SAVE_PREDICTION_RESULTS/
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

You can follow the official Docker guide for [installation of the Docker engine](https://docs.docker.com/engine/install/).

Next you need to install the `Docker compose` plugin that will allow us to spin up
multiple Docker containers. Instructions for that are [here](https://docs.docker.com/compose/install/linux/).

#### Building the SageMaker GraphStorm docker image
Follow [Build GraphStorm SageMaker docker image] (https://github.com/awslabs/graphstorm/docker/sagemaker) to build your own SageMaker GraphStorm docker image locally.

#### Creating the Docker compose file
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
builds the docker compose file for you, `generate_sagemaker_docker_compose.py`.
Note that the script uses the [PyYAML](https://pypi.org/project/PyYAML/) library.

This file has 4 required arguments that determine the Docker compose file that will be generated:

* `--num-instances`: The number of instances we want to launch.
This will determine the number of `algo-x` `services` entries our compose file ends up with.
* `--aws-access-key-id`: The AWS access key ID for accessing S3 data within docker
* `--aws-secret-access-key`: The AWS secret access key for accessing S3 data within docker.
* `--aws-session-token`: The AWS session toekn used for accessing S3 data within docker.

The rest of the arguments are passed on to `sagemaker_train.py` or `sagemaker_infer.py`:
* `--task-type`: Task type.
* `--graph-data-s3`: S3 location of the input graph.
* `--graph-name`: Name of the input graph.
* `--yaml-s3`: S3 location of yaml file for training and inference.
* `--custom-script`: Custom training script provided by a customer to run customer training logic. This should be a path to the python script within the docker image.
* `--output-emb-s3`: S3 location to store GraphStorm generated node embeddings. This is an inference only argument.
* `--output-prediction-s3`: S3 location to store prediction results. This is an inference only argument.

#### Docker compose for training
If you want to use Docker compose to testing training tasks.
You can use `generate_sagemaker_docker_compose.py` to generate compose file to run as following:

```
python3 generate_sagemaker_docker_compose.py \
    --aws-access-key <AWS_ACCESS_KEY> \
    --aws-secret-access-key <AWS_SECRET_ACCESS_KEY> \
    --image GRAPHSTORM_DOCKER_IMAGE \
    --num-instances 4 \
    --task-type node_classification \
    --graph-data-s3 s3://PATH_TO_GRAPH_DATA/ \
    --yaml-s3 s3://PATH_TO_YAML_FILE/\
    --model-artifact-s3 s3://PATH_TO_STORE_TRAINED_MODEL \
    --graph-name ogbn-arxiv \
    --num-layers 1 \
    --hidden-size 128 \
    --backend gloo \
    --batch-size 128 \
    --node-feat-name node:feat
```

As in the above example, if you want to pass other arguments to `sagemaker_train.py`,
you can simply append those arguments after `generate_sagemaker_docker_compose.py` arguments.
They will be passed on to the `sagemaker_train.py` script during execution.

The above will create a Docker compose file named `docker-compose-${task-type}-${num-instances}-train.yaml`, which we can then use to launch the job with (for example):

```bash
docker compose -f docker-compose-node_classification-4-train.yaml up
```

Running the above command will launch 4 instances of the image, configured with
the command and env vars that emulate a SageMaker execution environment and run
the `sagemaker_train.py` script. Note that the containers actually
interact with S3 so you would require valid AWS credentials to run.

#### Dcoker compose for inference
You can use `generate_sagemaker_docker_compose.py` to build docker compose file for testing inference tasks.
To create a compose file for inference you need to use the same arguments
as creating a compose file for the training task and pass another argument
to `generate_sagemaker_docker_compose.py` script, i.e., `--inference`.

```
python3 generate_sagemaker_docker_compose.py \
    --aws-access-key <AWS_ACCESS_KEY> \
    --aws-secret-access-key <AWS_SECRET_ACCESS_KEY> \
    --image GRAPHSTORM_DOCKER_IMAGE \
    --num-instances 4 \
    --task-type node_classification \
    --graph-data-s3 s3://PATH_TO_GRAPH_DATA/ \
    --yaml-s3 s3://PATH_TO_YAML_FILE/\
    --model-artifact-s3 s3://PATH_TO_SAVED_MODEL \
    --output-emb-s3 s3://PATH_TO_SAVE_NODE_EMBEDING \
    --output-prediction-s3 s3://PATH_TO_SAVE_PREDICTION_RESULT \
    --graph-name ogbn-arxiv \
    --num-layers 1 \
    --hidden-size 128 \
    --backend gloo \
    --batch-size 128 \
    --node-feat-name node:feat \
    --inference
```

As in the above example, if you want to pass other arguments to `sagemaker_infer.py`,
you can simply append those arguments after `generate_sagemaker_docker_compose.py` arguments.
They will be passed on to the `sagemaker_infer.py` script during execution.

The above will create a Docker compose file named `docker-compose-${task-type}-${num-instances}-infer.yaml`, which we can then use to launch the job with (for example):

```bash
docker compose -f docker-compose-node_classification-4-infer.yaml up
```

Running the above command will launch 4 instances of the image, configured with
the command and env vars that emulate a SageMaker execution environment and run
the `sagemaker_infer.py` script. Note that the containers actually
interact with S3 so you would require valid AWS credentials to run.
