# Running GraphStorm jobs on Amazon SageMaker
This submodule provides support to run GraphStorm graph construction, partitioning, training and inference on Amazon SageMaker.
To build a SageMaker compatible Docker image, please refer to [Build GraphStorm SageMaker docker image] (https://github.com/awslabs/graphstorm/docker/sagemaker).


## Launch SageMaker tasks
Use scripts under graphstorm.sagemaker.launch to launch SageMaker tasks.
Please make sure you already setup your SageMaker environment.
Please refer to [Amazon SageMaker service](https://aws.amazon.com/pm/sagemaker) for how to get access to Amazon SageMaker.

### Common task arguments

All SageMaker tasks share some common arguments that can be found in `<gs-root>/sagemaker/launch/common_parser.py`

* `image_url`: (required) The URI to the GraphStorm SageMaker image on ECR.
* `role`: (required) The SageMaker execution role that will be used to run the job.
* `instance-type`: The SageMaker instance type used to run the job.
* `instance-count`: The number of SageMaker instances used to run the job.
    For GConstruct jobs, the number is always 1.
* `region`: The region in which we will launch the job. Default is `us-west-2`, but ensure it matches the
    region of your image.
* `task-name`: A user-defined task name for the job.
* `sm-estimator-parameters`: Parameters that will be forwarded to the SageMaker Estimator/Processor.
    See the section `Passing additional arguments to the SageMaker Estimator/Processor` for details.
* `async-exection`: When this flag is set the job will run in async mode and return immediatelly.
    Otherwise the job will block and logs will be printed to the console.

### Launch GraphStorm graph construction using Amazon SageMaker service

#### Preparing the example dataset
We use the built-in acm dataset as an example.
You can generate the raw acm dataset in parquet format using the following instructions
(See [Use Your Own Data Tutorial](https://github.com/awslabs/graphstorm/wiki/tutorials-own-data#use-own-data) for more details):

```bash
cd ~/
git clone https://github.com/awslabs/graphstorm.git
GS_HOME=~/graphstorm/
cd $GS_HOME/examples
python3 acm_data.py --output-path acm_raw
```
The raw graph input data will be stored at ~/graphstorm/examples/acm_raw. The input configuration JSON is also generated and stored in the same path at  ~/graphstorm/examples/acm_raw/config.json.

#### Launch graph processing task
Before launching the task, you need to upload the raw acm dataset (i.e., /tmp/acm_raw) into S3.
```bash
aws s3 cp --recursive ~/graphstorm/examples/acm_raw s3://PATH_TO/acm/acm_raw/
```

Then, you can use the following command to launch a SageMaker graph construction task.

```bash
python3 launch/launch_gconstruct.py \
        --image-url <AMAZON_ECR_IMAGE_URI> \
        --region <region> \
        --entry-point run/gconstruct_entry.py \
        --role <ROLE_ARN> \
        --input-graph-s3 s3://PATH_TO/acm/ \
        --output-graph-s3 s3://PATH_TO/acm_output/ \
        --graph-name acm \
        --graph-config-file acm_raw/config.json \
        --sm-estimator-parameters "volume_size_in_gb=10"
```
The processed graph data is stored at s3://PATH_TO/acm_output/.

Note: The `--input-graph-s3` path will be mapped into `/opt/ml/processing/input` and used as the command working directory when launching the graph construction command.
The graph configuration file should be stored with the input graph data.
The argument `--graph-config-file` should be a relative path to `--input-graph-s3`.

### Launch graph partitioning task

If your data are in the [DGL chunked format](https://docs.dgl.ai/guide/distributed-preprocessing.html#specification)
you can perform distributed partitioning using SageMaker to prepare your data for distributed training.

```bash
python launch/launch_partition.py --graph-data-s3 ${DATASET_S3_PATH} \
    --num-parts ${NUM_PARTITIONS} --instance-count ${NUM_PARTITIONS} \
    --output-data-s3 ${OUTPUT_PATH} --instance-type ${INSTANCE_TYPE} \
    --image-url ${IMAGE_URI} --region ${REGION} \
    --role ${ROLE}  --entry-point "run/partition_entry.py" \
    --metadata-filename ${METADATA_FILE} \
    --log-level INFO --partition-algorithm ${ALGORITHM}
```

Running the above will take as input the dataset in chunked format from `${DATASET_S3_PATH}`
and create a DistDGL graph with `${NUM_PARTITIONS}` under the output path,
`${OUTPUT_PATH}`. Currently we only support `random` as the partitioning algorithm.

### Launch GraphStorm training using Amazon SageMaker service

#### Preparing training data and training task config.

We use the built-in ogbn-arxiv dataset as an example.
First you need to partition the graph dataset by following the instructions:
```bash
cd ~/
git clone https://github.com/awslabs/graphstorm.git
GS_HOME=~/graphstorm/
python3 $GS_HOME/tools/partition_graph.py --dataset ogbn-arxiv \
                                          --filepath /tmp/ogbn-arxiv-nc/ \
                                          --num_parts 2 \
                                          --output /tmp/ogbn_arxiv_nc_2p
```
The partitioned graph will be stored at /tmp/ogbn_arxiv_nc_2p.

#### Launch train task
Before launching the task, you need to upload the partitioned graph (i.e., /tmp/ogbn_arxiv_nc_2p) into S3.
You also need to upload the yaml config file into S3.
You can find the example yaml file in https://github.com/awslabs/graphstorm/blob/main/training_scripts/gsgnn_np/arxiv_nc.yaml.
```bash
aws s3 cp --recursive /tmp/ogbn_arxiv_nc_2p s3://PATH_TO/ogbn_arxiv_nc_2p/
aws s3 cp PATH_TO/arxiv_nc.yaml s3://PATH_TO_TRAINING_CONFIG/arxiv_nc.yaml
```

Then, you can use the following command to launch a SageMaker training task.
```bash
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

If you want to resume a saved model checkpoint to do model fine-tuning you can pass
the S3 address of the model checkpoint through the ``--model-checkpoint-to-load``
argument. For example by passing ``--model-checkpoint-to-load s3://mag-model/epoch-2/``,
GraphStorm will initialize the model parameters with the model checkpoint stored in ``s3://mag-model/epoch-2/``.

Please note `save_embed_path` and `save_prediction_path` must be disabled, i.e., set to 'None' when using SageMaker.
They only work with shared file system while SageMaker solution does not support using shared file system now.


### Launch inference task using built-in inference script
Inference task can use the same graph as training task. You can also run inference on a new graph.
In this example, we will use the same graph.

You can use the following command to launch a SageMaker offline inference task.
```bash
cd $GS_HOME/sagemaker/
python3 launch/launch_infer.py \
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

```bash
aws s3 ls s3://PATH_TO_SAVE_GENERATED_NODE_EMBEDDING/
aws s3 ls s3://PATH_TO_SAVE_PREDICTION_RESULTS/
```

### Passing additional arguments to the SageMaker Estimator/Processor

Sometimes you might want to pass additional arguments to the constructor of the SageMaker
Estimator/Processor object we use to launch SageMaker tasks, e.g. to set a max runtime, or
set a VPC configuration. Our launch scripts support forwarding arguments to the base class
object through a `kwargs` dictionary.

To pass additional `kwargs` directly to the Estimator/Processor constructor, you can use
the `--sm-estimator-parameters` argument, providing a string of space-separated arguments
(enclosed in double quotes `"` to ensure correct parsing) and the
format `<argname>=<value>` for each argument.

`<argname>` needs to be a valid SageMaker Estimator/Processor
argument name and `<value>` a value that can be parsed as a Python literal, **without
spaces**.

For example, to pass a specific max runtime, subnet list, and enable inter-container
traffic encryption for a train, inference, or partition job you'd use:

```bash
python3 launch/launch_[infer|train|partition] \
    <other arugments> \
    --sm-estimator-parameters "max_run=3600 volume_size=100 encrypt_inter_container_traffic=True subnets=['subnet-1234','subnet-4567']"
```

Notice how we don't include any spaces in `['subnet-1234','subnet-4567']` to ensure correct parsing of the list.

The train, inference and partition scripts launch SageMaker Training jobs that rely on the `Estimator` base class:
For a full list of `Estimator` parameters see:
https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.EstimatorBase

The GConstruct job will launch a SageMaker Processing job that relies on the `Processor` base class, so its
arguments are different, e.g. `volume_size_in_gb` for the `Processor` vs. `volume_size` for the `Estimator`.

So the above example would become:

```bash
python3 launch/launch_gconstruct \
    <other arugments> \
    --sm-estimator-parameters "max_runtime_in_seconds=3600 volume_size_in_gb=100"
```


For a full list of `Processor` parameters see:
https://sagemaker.readthedocs.io/en/stable/api/training/processing.html



---

## Test GraphStorm SageMaker runs locally with Docker compose
This section describes how to launch Docker compose jobs that emulate a SageMaker
training execution environment that can be used to test GraphStorm model training
and inference using SageMaker.

### TLDR

1. Install Docker and the Docker compose plugin: https://docs.docker.com/compose/install/linux/
2. Clone GraphStorm `git clone git@github.com:awslabs/graphstorm.git`.
3. Build the SageMaker GraphStorm Docker image using `graphstorm/docker/build_docker_sagemaker.sh`.
4. Ensure you have valid AWS credentials to access the S3 paths involved.
5. Generate a docker compose file:
```bash
python local/generate_sagemaker_docker_compose.py --image graphstorm:sm  \
    --graph-name ${GRAPH_NAME} --num-instances ${NUM_INSTANCES} \
    --graph-data-s3 ${DATASET_S3_PATH} --region ${REGION} \
    training \
    --task-type ${TASK_TYPE} --train-yaml-s3 ${TRAIN_YAML_S3} \
    --model-artifact-s3 ${OUTPUT_S3_PATH} $GSF_ARGS
```
6. Launch the job using docker compose:
```bash
docker compose -f "docker-compose-train-${GRAPH_NAME}-${NUM_INSTANCES}workers-${TASK_TYPE}.yml" up
```

### Getting started
If you’ve never worked with Docker compose before the official description provides a great intro:

> Compose is a tool for defining and running multi-container Docker applications.
With Compose, you use a YAML file to configure your application’s services.
Then, with a single command, you create and start all the services from your configuration.

We will use this capability to launch multiple worker instances locally, that will
be configured to “look like” an Amazon SageMaker training instance and communicate over a
virtual network created by Docker compose. This way our test environment will be
as close to a real SageMaker distributed job as we can get, without needing to
launch SageMaker jobs, or launch and configure multiple EC2 instances when
developing features. Once we are done debugging locally we can launch our
SageMaker jobs with the full data on AWS using the instructions in the previous
section.

### Prerequisite 1: Launch or re-use a GPU EC2 instance

As we will be running multiple heavy containers is one machine we recommend using
a capable Linux-based machine equiped with GPUs. We recommend using an EC2 Linux instance
with at least 32GB of RAM, with an execution role configured to access the neccessary files
on S3. See the [EC2 documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html#working-with-iam-roles) for how to launch an instance with a role attached.


### Prerequisite 2: Install Docker and docker compose

You can follow the official Docker guide for [installation of the Docker engine](https://docs.docker.com/engine/install/).

Next you need to install the `Docker compose` plugin that will allow us to spin up
multiple Docker containers. Instructions for that are [here](https://docs.docker.com/compose/install/linux/).

### Building the SageMaker GraphStorm docker image
Follow [Build GraphStorm SageMaker docker image] (https://github.com/awslabs/graphstorm/docker/sagemaker) to build your own SageMaker GraphStorm docker image locally. After building there should be an image tagged `sagemaker:sm` available locally
when you run `docker images graphstorm`.

### Creating the Docker compose file
A Docker compose file is a YAML file that tells Docker which containers to spin up and how to configure them.
To launch the services with a Docker compose file, we can use `docker compose -f docker-compose.yaml up`.
This will launch the container and execute its entry point.

To emulate a SageMaker distributed execution environment based on the image you built previously you would need a Docker compose file that looks like this:
```yaml
version: '3.7'

networks:
  gsf:
    name: gsf-network

services:
  algo-1:
    image: graphstorm:sm
    container_name: algo-1
    hostname: algo-1
    networks:
      - gsf
    command: 'THIS_WILL_RUN_INSIDE_THE_CONTAINER'
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
Note that the script requires the [PyYAML](https://pypi.org/project/PyYAML/) library to be installed (`pip install pyyaml`).

This file has 4 required keyword arguments that determine the Docker compose file that will be generated:

* `--num-instances`: The number of instances we want to launch.
This will determine the number of `algo-x` `services` entries our compose file ends up with.
* `--graph-name`: Name of the input graph.
* `--graph-data-s3`: S3 location of the input graph.
* `--region`: The region of the S3 bucket we will use for input and output. We assume both the input and ouput buckets were created
in the same region.

The top-level optional arguments for the script are:
* `--image`: The name of the local image we'll be using to launch tasks. Defaults to `graphstorm:sm`.
* `--aws-access-key-id`: The AWS access key ID for accessing S3 data within docker
* `--aws-secret-access-key`: The AWS secret access key for accessing S3 data within docker.
* `--aws-session-token`: The AWS session token used for accessing S3 data within docker.
* `--log-level`: The log level for the script.
Note that you only need to provide the credentials if you don't have them already configured
on the host.

With the above ketwords arguments set, we use the positional argument `action` to choose one
of `partitioning`, `training`, `inference`, e.g.:

```
python local/generate_sagemaker_docker_compose.py --graph-name ${GRAPH_NAME} --num-instances ${NUM_INSTANCES} \
    --graph-data-s3 ${DATASET_S3_PATH} --region ${REGION} \
    training \
    # Action-specific arguments go here
```

We explain the specific arguments for each action in the following sections.

### Providing AWS credentials

Note that the containers actually interact with S3 so you would require valid AWS credentials to run
the above.

If you already have configured AWS credentials, for example through the AWS CLI, or by using
an execution role on an EC2 instance, you will not need to provide AWS credentials.

Otherwise you can use the optional `--aws-access-key-id`, `--aws-secret-access-key`, and
`--aws-session-token` arguments to provide _temporary_  AWS credentials to the generated Docker compose files.

We recommend that you rely on either EC2 IAM roles with specific policies, or follow the
[recommendations from the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-authentication.html)
for credential management.


### Docker compose for partitioning

If your data is in DGL's chunked format you can use Docker compose to test distributed partitioning algorithms.
You can use `generate_sagemaker_docker_compose.py` to generate compose file to run as following:

```bash
NUM_INSTANCES=4
NUM_PARTITIONS=4
python local/generate_sagemaker_docker_compose.py partitioning --image graphstorm:sm \
    --graph-name ${GRAPH_NAME} --num-instances ${NUM_INSTANCES} \
    --graph-data-s3 ${DATASET_S3_PATH} --region ${REGION} \
    --num-parts ${NUM_PARTITIONS} --metadata-filename ${METADATA_FILE} --partition-algorithm ${ALGORITHM} \
    --output-data-s3 ${OUTPUT_S3_PATH}
```

For the `partitioning` action the available args are:

* `--num-parts`: The number of partitions to generate.
* `--output-data-s3`: The S3 prefix under which we will generate the output partitions.
* `--skip-partitioning`: If partition assignments already exist under the output, we can skip the
    partitioning step and only generate the DistDGL objects.
* `--metadata-filename`: The JSON file that describes the data in DistDGL's chunked data format.
* `--partition-algorithm`: The partition algorithm to use. Currently only supports `random`.


The script will create a Docker compose file named
`docker-compose-partitioning-${GRAPH_NAME}-${NUM_ISNTANCES}workers-random-${NUM_PARTITIONS}parts.yaml`,
which we can then use to launch the job with (for example):

```bash
docker compose -f docker-compose-partitioning-acm-4workers-random-4parts.yaml up
```


### Docker compose for training
If you want to use Docker compose to test training tasks before launching SageMaker jobs you can use
Docker compose to test distributed training locally:

```bash
GSF_ARGS="--num-layers 1 --hidden-size 128 --backend gloo --batch-size 128"
python local/generate_sagemaker_docker_compose.py training --image graphstorm:sm  \
    --graph-name ${GRAPH_NAME} --num-instances ${NUM_INSTANCES} \
    --graph-data-s3 ${DATASET_S3_PATH} --region ${REGION} \
    --task-type ${TASK_TYPE} --train-yaml-s3 ${TRAIN_YAML_S3} \
    --model-artifact-s3 ${OUTPUT_S3_PATH} $GSF_ARGS
```

For the `training` action the available args are:

* `--task-type`: Task type from `node_classification`, `node_regression`, `edge_classification`, `edge_regression`, `link_prediction`.
* `--train-yaml-s3`: S3 location of yaml file for training.
* `--model-artifact-s3`: The S3 prefix under which we will generate the model artifacts.
* `--custom-script`: Custom training script provided by a customer to run customer training logic. This should be a path to the python script within the docker image.

As with launching jobs on SageMaker, if you want to pass other arguments to `sagemaker_train.py`,
you can simply append those arguments after `generate_sagemaker_docker_compose.py` arguments.
Here we use the `GSF_ARGS` variable to pass all GSF args in a group.
They will be passed on to the `sagemaker_train.py` script during execution.

The script will create a Docker compose file named
`docker-compose-training-${GRAPH_NAME}-${NUM_INSTANCES-instances}workers-${TASK_TYPE}.yaml`,
which we can then use to launch the job with (for example):

```bash
docker compose -f docker-compose-training-4workers-node_classification.yaml up
```

Running the above command will launch 4 instances of the image, configured with
the command and env vars that emulate a SageMaker execution environment and run
the `sagemaker_train.py` script.

### Docker compose for inference
You can use `generate_sagemaker_docker_compose.py` to build docker compose file for testing inference tasks.
To generate a compose YAML for inference you'd use:

```bash
GSF_ARGS="--num-layers 1 --hidden-size 128 --backend gloo --batch-size 128"
python local/generate_sagemaker_docker_compose.py inference --image graphstorm:sm \
    --graph-name ${GRAPH_NAME} --num-instances ${NUM_INSTANCES} \
    --graph-data-s3 ${DATASET_S3_PATH} --region ${REGION} \
    --task-type ${TASK_TYPE} \
    --model-artifact-s3 ${MODEL_ARTIFACT_S3} --infer-yaml-s3 ${INFER_YAML_S3} \
    --output-emb-s3 ${OUTPUT_S3_PATH}/embeddings --output-prediction-s3 ${OUTPUT_S3_PATH}/predictions \
    $GSF_ARGS
```

For the `inference` action the available args are:

* `--infer-yaml-s3`: S3 location of yaml file for inference.
* `--task-type`: Task type from `node_classification`, `node_regression`, `edge_classification`, `edge_regression`, `link_prediction`.
* `--model-artifact-s3`: The S3 prefix from which we will load the model artifacts.
* `--output-emb-s3`: S3 location to store GraphStorm generated node embeddings. This is an inference only argument.
* `--output-prediction-s3`: S3 location to store prediction results. This is an inference only argument.
* `--custom-script`: Custom training script provided by a customer to run customer training logic. This should be a path to the python script within the docker image.

As in the training example, if you want to pass other arguments to `sagemaker_infer.py`,
you can simply append those arguments after `generate_sagemaker_docker_compose.py` arguments.
They will be passed on to the `sagemaker_infer.py` script during execution.

The script will create a Docker compose file named
`docker-compose-inference-${num-NUM_INSTANCES}workers-${TASK_TYPE}.yaml`,
which we can then use to launch the job with (for example):

```bash
docker compose -f docker-compose-inference-4workers-node_classification.yaml up
```

Running the above command will launch 4 instances of the image, configured with
the command and env vars that emulate a SageMaker execution environment and run
the `sagemaker_infer.py` script.

### Troubleshooting

When running actions in sequence (e.g. partitioning, training, inference) remember to clean up containers between actions either by running
`docker compose -f docker-compose-inference-4workers-node_classification.yaml down` or
`docker container prune -f`. This will avoid the issue of a container name already being taken (containers share the algo-* name).