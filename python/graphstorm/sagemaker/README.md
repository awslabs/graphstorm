# Amazon SageMaker Support
This submodule provides support to run GraphStorm training and inference on Amazon SageMaker. To build a SageMaker compatible docker image, please refer to [Build GraphStorm SageMaker docker image] (https://github.com/awslabs/graphstorm/docker/sagemaker).

## Launch SageMaker tasks
Use scripts under graphstorm.sagemaker.launch to launch SageMaker tasks. Please make sure you already setup your SageMaker environments. Please refer to [Amazon SageMaker service](https://aws.amazon.com/pm/sagemaker) for how to get access to Amazon SageMaker.


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
python3 -m graphstorm.sagemaker.local.generate_sagemaker_train_docker_compose --num-instances $NUM_INSTANCES --input-data $DATASET_S3_PATH --output-data-s3 "s3://${OUTPUT_BUCKET}/test/${DATASET_NAME}${PATH_SUFFIX}/${NUM_INSTANCES}x-${INSTANCE_TYPE}/" --region $REGION
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
builds the docker compose file for your, `generate_sagemaker_train_docker_compose.py`.
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
* `--enable-bert`: Whether enable cotraining Bert with GNN
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


### Launch GraphStorm training/inference using Amazon SageMaker service

### Launch train task using built-in training script

Preparing training data and training task config
```
```

Launch train task
```
```


### Launch train task using user-defined training script

Preparing training data and training task config
```
```

Launch train task
```
```


### Launch inference task using built-in inference script

Preparing inference data and inference task config
```
```

Launch inference task
```
```


### Launch inference task using user-defined inference script

Preparing inference data and inference task config
```
```

Launch inference task
```
```