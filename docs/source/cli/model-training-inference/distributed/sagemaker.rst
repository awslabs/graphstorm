.. _distributed-sagemaker:

Model Training and Inference on on SageMaker
=============================================

GraphStorm can run on Amazon Sagemaker to leverage SageMaker's ML DevOps capabilities.

Prerequisites
-----------------
In order to use GraphStorm on Amazon SageMaker, users need to have AWS access to the following AWS services.

- **SageMaker service**. Please refer to `Amazon SageMaker service <https://aws.amazon.com/pm/sagemaker/>`_ for how to get access to Amazon SageMaker.
- **Amazon ECR**. Please refer to `Amazon Elastic Container Registry service <https://aws.amazon.com/ecr/>`_ for how to get access to Amazon ECR.
- **S3 service**. Please refer to `Amazon S3 service <https://aws.amazon.com/s3/>`_ for how to get access to Amazon S3.
- **SageMaker Framework Containers**. Please follow `AWS Deep Learning Containers guideline <https://github.com/aws/deep-learning-containers>`_ to get access to the image.
- **Amazon EC2** (optional). Please refer to `Amazon EC2 service <https://aws.amazon.com/ec2/>`_ for how to get access to Amazon EC2.

Setup GraphStorm SageMaker Docker Image
----------------------------------------------
GraphStorm uses SageMaker's **BYOC** (Bring Your Own Container) mode. Therefore, before launching GraphStorm on SageMaker, there are two steps required to setup a GraphStorm SageMaker Docker image.

.. _build_sagemaker_docker:

Step 1: Build a SageMaker-compatible Docker image
...................................................

.. note::
    * Please make sure your account has access key (AK) and security access key (SK) configured to authenticate accesses to AWS services.
    * For more details of Amazon ECR operation via CLI, users can refer to the `Using Amazon ECR with the AWS CLI document <https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html>`_.

First, in a Linux machine, configure a Docker environment by following the `Docker documentation <https://docs.docker.com/get-docker/>`_ suggestions.

In order to use the SageMaker base Docker image, users need to use the following command to authenticate to pull SageMaker images.

.. code-block:: bash

    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

Then, clone GraphStorm source code, and build a GraphStorm SageMaker compatible Docker image from source with commands:

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git

    cd /path-to-graphstorm/docker/

    bash /path-to-graphstorm/docker/build_docker_sagemaker.sh /path-to-graphstorm/ <DEVICE_TYPE> <IMAGE_NAME> <IMAGE_TAG>

The ``build_docker_sagemaker.sh`` script takes four arguments:

1. **path-to-graphstorm** (**required**), is the absolute path of the ``graphstorm`` folder, where you cloned the GraphStorm source code. For example, the path could be ``/code/graphstorm``.
2. **DEVICE_TYPE** (optional), is the intended device type of the to-be built Docker image. There are two options: ``cpu`` for building CPU-compatible images, and ``gpu`` for building Nvidia GPU-compatible images. Default is ``gpu``.
3. **IMAGE_NAME** (optional), is the assigned name of the to-be built Docker image. Default is ``graphstorm``.

.. warning::
    In order to upload the GraphStorm SageMaker Docker image to Amazon ECR, users need to define the <IMAGE_NAME> to include the ECR URI string, **<AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/**, e.g., ``888888888888.dkr.ecr.us-east-1.amazonaws.com/graphstorm``.

4. **IMAGE_TAG** (optional), is the assigned tag name of the to-be built Docker image. Default is ``sm-<DEVICE_TYPE>``,
   that is, ``sm-gpu`` for GPU images, ``sm-cpu`` for CPU images.

Once the ``build_docker_sagemaker.sh`` command completes successfully, there will be a Docker image, named ``<IMAGE_NAME>:<IMAGE_TAG>``,
such as ``888888888888.dkr.ecr.us-east-1.amazonaws.com/graphstorm:sm-gpu``, in the local repository, which could be listed by running:

.. code-block:: bash

    docker image ls

.. _upload_sagemaker_docker:

Step 2: Upload Docker Images to Amazon ECR Repository
.......................................................
Because SageMaker relies on Amazon ECR to access customers' own Docker images, users need to upload Docker images built in the Step 1 to their own ECR repository.

The following command will authenticate the user account to access to user's ECR repository via AWS CLI.

.. code-block:: bash

    aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com

Please replace the `<REGION>` and `<AWS_ACCOUNT_ID>` with your own account information and be consistent with the values used in the **Step 1**.

In addition, users need to create an ECR repository at the specified `<REGION>` with the name as `<IMAGE_NAME>` **WITHOUT** the ECR URI string, e.g., ``graphstorm``.

And then use the below command to push the built GraphStorm Docker image to users' own ECR repository.

.. code-block:: bash

    docker push <IMAGE_NAME>:<IMAGE_TAG>

Please replace the `<IMAGE_NAME>` and `<IMAGE_TAG>` with the actual Docker image name and tag, e.g., ``888888888888.dkr.ecr.us-east-1.amazonaws.com/graphstorm:sm-gpu``.

Run GraphStorm on SageMaker
----------------------------
There are two ways to run GraphStorm on SageMaker.

* **Run with Amazon SageMaker service**. In this way, users will use GraphStorm's tools to submit SageMaker API calls, which request SageMaker services to start new SageMaker training or inference instances that run GraphStorm code. Users can submit the API calls on a properly configured machine without GPUs (e.g., C5.xlarge). This is the formal way to run GraphStorm experiments on large graphs and to deploy GraphStorm on SageMaker for production environment.
* **Run with Docker Compose in a local environment**. In this way, users do not call the SageMaker service, but use Docker Compose to run SageMaker locally in a Linux instance that has GPUs. This is mainly for model developers and testers to simulate running GraphStorm on SageMaker.

Run GraphStorm with Amazon SageMaker service
..............................................
To run GraphStorm with the Amazon SageMaker service, users should set up an instance with the SageMaker library installed and GraphStorm's SageMaker tools copied.

1. Use the below command to install SageMaker.

.. code-block:: bash

    pip install sagemaker

2. Copy GraphStorm SageMaker tools. Users can clone the GraphStorm repository using the following command or copy the `sagemaker folder <https://github.com/awslabs/graphstorm/tree/main/sagemaker>`_ to the instance.

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git

Prepare graph data
`````````````````````
Unlike GraphStorm's :ref:`Standalone mode<quick-start-standalone>` and :ref:`the Distributed mode<distributed-cluster>`, which rely on local disk or shared file system to store the partitioned graph, SageMaker utilizes Amazon S3 as the shared data storage for distributing partitioned graphs and the configuration YAML file.

This tutorial uses the same three-partition OGB-MAG graph and the Link Prediction task as those introduced in the :ref:`Partition a Graph<partition-a-graph>` section of the :ref:`Use GraphStorm in a Distributed Cluster<distributed-cluster>` tutorial. After generating the partitioned OGB-MAG graphs, use the following commands to upload them and the configuration YAML file to an S3 bucket.

.. code-block:: bash

    aws s3 cp --recursive /data/ogbn_mag_lp_3p s3://<PATH_TO_DATA>/ogbn_mag_lp_3p
    aws s3 cp /graphstorm/training_scripts/gsgnn_lp/mag_lp.yaml s3://<PATH_TO_TRAINING_CONFIG>/mag_lp.yaml

Please replace `<PATH_TO_DATA>` and `<PATH_TO_TRAINING_CONFIG>` with your own S3 bucket URI.

Launch training
```````````````````
Launching GraphStorm training on SageMaker is similar as launching in the :ref:`Standalone mode<quick-start-standalone>` and :ref:`the Distributed mode<distributed-cluster>`, except for three diffences:

* The launch commands are located in the ``graphstorm/sagemaker`` folder, and
* Users need to provide AWS service-related information in the command.
* All paths for saving models, embeddings, and prediction results should be specified as S3 locations using the S3 related arguments.

Users can use the following commands to launch a GraphStorm Link Prediction training job with the OGB-MAG graph.

.. code-block:: bash

    cd /path-to-graphstorm/sagemaker/

    python3 launch/launch_train.py \
            --image-url <AMAZON_ECR_IMAGE_URI> \
            --region <REGION> \
            --entry-point run/train_entry.py \
            --role <ROLE_ARN> \
            --instance-count 3 \
            --graph-data-s3 s3://<PATH_TO_DATA>/ogbn_mag_lp_3p \
            --yaml-s3 s3://<PATH_TO_TRAINING_CONFIG>/mag_lp.yaml \
            --model-artifact-s3 s3://<PATH_TO_SAVE_TRAINED_MODEL>/ \
            --graph-name ogbn-mag \
            --task-type link_prediction \
            --lp-decoder-type dot_product \
            --num-layers 1 \
            --fanout 10 \
            --hidden-size 128 \
            --backend gloo \
            --batch-size 128

Please replace `<AMAZON_ECR_IMAGE_URI>` with the `<IMAGE_NAME>:<IMAGE_TAG>` that are uploaded in the Step 2, e.g., ``888888888888.dkr.ecr.us-east-1.amazonaws.com/graphstorm:sm``, replace the `<REGION>` with the region where ECR image repository is located, e.g., ``us-east-1``, and replace the `<ROLE_ARN>` with your AWS account ARN that has SageMaker execution role, e.g., ``"arn:aws:iam::<ACCOUNT_ID>:role/service-role/AmazonSageMaker-ExecutionRole-20220627T143571"``.

Because we are using a three-partition OGB-MAG graph, we need to set the ``--instance-count`` to 3 in this command.

The trained model artifact will be stored in the S3 location provided through the ``--model-artifact-s3`` argument. You can use the following command to check the model artifacts after the training completes.

If you want to resume a saved model checkpoint to do model fine-tuning you can pass
the S3 address of the model checkpoint through the ``--model-checkpoint-to-load``
argument. For example by passing ``--model-checkpoint-to-load s3://mag-model/epoch-2/``,
GraphStorm will initialize the model parameters with the model checkpoint stored in ``s3://mag-model/epoch-2/``.

.. code-block:: bash

    aws s3 ls s3://<PATH_TO_SAVE_TRAINED_MODEL>/

Launch inference
`````````````````````
Users can use the following command to launch a GraphStorm Link Prediction inference job on the OGB-MAG graph.

.. code-block:: bash

    python3 launch/launch_infer.py \
            --image-url <AMAZON_ECR_IMAGE_URI> \
            --region <REGION> \
            --entry-point run/infer_entry.py \
            --role <ROLE_ARN> \
            --instance-count 3 \
            --graph-data-s3 s3://<PATH_TO_DATA>/ogbn_mag_lp_3p \
            --yaml-s3 s3://<PATH_TO_TRAINING_CONFIG>/mag_lp.yaml \
            --model-artifact-s3 s3://<PATH_TO_SAVE_TRAINED_MODEL>/ \
            --raw-node-mappings-s3 s3://<PATH_TO_DATA>/ogbn_mag_lp_3p/raw_id_mappings \
            --output-emb-s3 s3://<PATH_TO_SAVE_GENERATED_NODE_EMBEDDING>/ \
            --output-prediction-s3 s3://<PATH_TO_SAVE_PREDICTION_RESULTS> \
            --graph-name ogbn-mag \
            --task-type link_prediction \
            --num-layers 1 \
            --fanout 10 \
            --hidden-size 128 \
            --backend gloo \
            --batch-size 128

.. note::

    * Different from the training command's argument, in the inference command, the value of the ``--model-artifact-s3`` argument needs to be path to a saved model. By default, it is stored under an S3 path with specific training epoch or epoch plus iteration number, e.g., ``s3://models/epoch-0-iter-999``, where the trained model artifacts were saved.
    * If ``--raw-node-mappings-s3`` is not provided, it will be default to the ``{graph-data-s3}/raw_id_mappings``. The expected graph mappings files should be ``node_mapping.pt``, ``edge_mapping.pt`` and parquet files under ``raw_id_mappings``. They record the mapping between original node and edge ids in the raw data files and the ids of nodes and edges in the Graph Node ID space. These files are created during graph construction by either GConstruct or GSProcessing.

As the outcomes of the inference command, the generated node embeddings will be uploaded to ``s3://<PATH_TO_SAVE_GENERATED_NODE_EMBEDDING>/``. For node classification/regression or edge classification/regression tasks, users can use ``--output-prediction-s3`` to specify the saving locations of prediction results.

Users can use the following commands to check the corresponding outputs:

.. code-block:: bash

    aws s3 ls s3://<PATH_TO_SAVE_GENERATED_NODE_EMBEDDING>/
    aws s3 ls s3://<PATH_TO_SAVE_PREDICTION_RESULTS>/

Launch graph partitioning task
```````````````````````````````
If your data are in the `DGL chunked
format <https://docs.dgl.ai/guide/distributed-preprocessing.html#specification>`_
you can perform distributed partitioning using SageMaker to prepare your
data for distributed training.

.. code:: bash

   python launch/launch_partition.py \
       --graph-data-s3 ${DATASET_S3_PATH} \
       --num-parts ${NUM_PARTITIONS} \
       --instance-count ${NUM_PARTITIONS} \
       --output-data-s3 ${OUTPUT_PATH} \
       --instance-type ${INSTANCE_TYPE} \
       --image-url ${IMAGE_URI} \
       --region ${REGION} \
       --role ${ROLE}  \
       --entry-point "run/partition_entry.py" \
       --metadata-filename ${METADATA_FILE} \
       --log-level INFO \
       --partition-algorithm ${ALGORITHM}

Running the above will take the dataset in chunked format
from ``${DATASET_S3_PATH}`` as input and create a DistDGL graph with
``${NUM_PARTITIONS}`` under the output path, ``${OUTPUT_PATH}``.
Currently we only support ``random`` as the partitioning algorithm.

Passing additional arguments to the SageMaker
`````````````````````````````````````````````
Sometimes you might want to pass additional arguments to the constructor
of the SageMaker Estimator/Processor object that we use to launch SageMaker
tasks, e.g. to set a max runtime, or set a VPC configuration. Our launch
scripts support forwarding arguments to the base class object through a
``kwargs`` dictionary.

To pass additional ``kwargs`` directly to the Estimator/Processor
constructor, you can use the ``--sm-estimator-parameters`` argument,
providing a string of space-separated arguments (enclosed in double
quotes ``"`` to ensure correct parsing) and the format
``<argname>=<value>`` for each argument.

``<argname>`` needs to be a valid SageMaker Estimator/Processor argument
name and ``<value>`` is a value that can be parsed as a Python literal,
**without spaces**.

For example, to pass a specific max runtime, subnet list, and enable
inter-container traffic encryption for a train, inference, or partition
job you'd use:

.. code:: bash

   python3 launch/launch_[infer|train|partition] \
       <other arugments> \
       --sm-estimator-parameters "max_run=3600 volume_size=100 encrypt_inter_container_traffic=True subnets=['subnet-1234','subnet-4567']"

Notice how we don't include any spaces in
``['subnet-1234','subnet-4567']`` to ensure correct parsing of the list.

The train, inference and partition scripts launch SageMaker Training
jobs that rely on the ``Estimator`` base class: For a full list of
``Estimator`` parameters see the `SageMaker Estimator documentation.
<https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.EstimatorBase>`_

The GConstruct job will launch a SageMaker Processing job that relies on
the ``Processor`` base class, so its arguments are different,
e.g. ``volume_size_in_gb`` for the ``Processor`` vs. ``volume_size`` for
the ``Estimator``. For a full list of ``Processor`` parameters, see the `SageMaker Processor documentation.
<https://sagemaker.readthedocs.io/en/stable/api/training/processing.html>`_

Using ``Processor`` arguments the above example would become:

.. code:: bash

   python3 launch/launch_gconstruct \
       <other arugments> \
       --sm-estimator-parameters "max_runtime_in_seconds=3600 volume_size_in_gb=100"


Run GraphStorm SageMaker with Docker Compose
..............................................
This section describes how to launch Docker compose jobs that emulate a SageMaker training execution environment. This can be used to develop and test GraphStorm model training and inference on SageMaker locally.

If users have never worked with Docker compose before the official description provides a great intro:

.. hint::

    Compose is a tool for defining and running multi-container Docker applications. With Compose, you use a YAML file to configure your application's services. Then, with a single command, you create and start all the services from your configuration.

We will use this capability to launch multiple worker instances locally, that will be configured to “look like” a SageMaker training instance and communicate over a virtual network created by Docker Compose. This way our test environment will be as close to a real SageMaker distributed job as we can get, without needing to launch SageMaker jobs, or launch and configure multiple EC2 instances when developing features.

Get Started
`````````````
To run GraphStorm SageMaker with Docker Compose, we need to set up a local Linux instance with the following contents.

1. Use the below command to install SageMaker.

.. code-block:: bash

    pip install sagemaker

2. Clone GraphStorm code.

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git

3. Setup GraphStorm in the PYTHONPATH variable.

.. code-block:: bash

    export PYTHONPATH=/PATH_TO_GRAPHSTORM/python:$PYTHONPATH

4. Build a SageMaker compatible Docker image following the :ref:`Step 1 <build_sagemaker_docker>`.

5. Follow the `Docker Compose <https://docs.docker.com/compose/install/linux/>`_ documentation to install Docker Compose.

Generate a Docker Compose file
`````````````````````````````````
A Docker Compose file is a YAML file that tells Docker which containers to spin up and how to configure them. To launch the services with a Docker Compose file, we can use ``docker compose -f docker-compose.yaml up``. This will launch the container and execute its entry point.

To emulate a SageMaker distributed execution environment based on the previously built Docker image (suppose the docker image is named ``graphstorm:sm``), you would need a Docker Compose file that resembles the following:

.. code-block:: yaml

    version: '3.7'

    networks:
    gfs:
        name: gsf-network

    services:
    algo-1:
        image: graphstorm:sm
        container_name: algo-1
        hostname: algo-1
        networks:
        - gsf
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

Some explanation on the above elements (see the `official docs <https://docs.docker.com/compose/compose-file/>`_ for more details):

* **image**: Specifies the Docker image that will be used for launching the container. In this case, the image is ``graphstorm:sm``, which should correspond to the previously built Docker image.
* **environment**: Sets the environment variables for the container.
* **command**: Specifies the entry point, i.e., the command that will be executed when the container launches. In this case, /path/to/entrypoint.sh is the command that will be executed.

To help users generate yaml file automatically, GraphStorm provides a Python script, ``generate_sagemaker_docker_compose.py``, that builds the docker compose file for users.

.. Note:: The script uses the `PyYAML <https://pypi.org/project/PyYAML/>`_ library. Please use the below commnd to install it.

    .. code-block:: bash

        pip install pyyaml

This Python script has 4 required arguments that determine the Docker Compose file that will be generated:

* **--aws-access-key-id**: The AWS access key ID for accessing S3 data within docker
* **--aws-secret-access-key**: The AWS secret access key for accessing S3 data within docker.
* **--aws-session-token**: The AWS session toekn used for accessing S3 data within docker.
* **--num-instances**: The number of instances we want to launch. This will determine the number of algo-x services entries our compose file ends up with.

The rest of the arguments are passed on to ``sagemaker_train.py`` or ``sagemaker_infer.py``:

* **--task-type**: Task type.
* **--graph-data-s3**: S3 location of the input graph.
* **--graph-name**: Name of the input graph.
* **--yaml-s3**: S3 location of yaml file for training and inference.
* **--custom-script**: Custom training script provided by customers to run customer training logic. This should be a path to the Python script within the Docker image.
* **--output-emb-s3**: S3 location to store GraphStorm generated node embeddings. This is an inference only argument.
* **--output-prediction-s3**: S3 location to store prediction results. This is an inference only argument.

Run GraphStorm on Docker Compose for Training
```````````````````````````````````````````````
First, use the following command to generate a Compose YAML file for the Link Prediction training on OGB-MAG graph.

.. code-block:: bash

    python3 generate_sagemaker_docker_compose.py \
            --aws-access-key <<AWS_ACCESS_KEY>> \
            --aws-secret-access-key <AWS_SECRET_ACCESS_KEY> \
            --aws-session-token <AWS_SESSION_TOKEN> \
            --num-instances 3 \
            --image <GRAPHSTORM_DOCKER_IMAGE> \
            --graph-data-s3 s3://<PATH_TO_DATA>/ogbn_mag_lp_3p \
            --yaml-s3 s3://<PATH_TO_TRAINING_CONFIG>/map_lp.yaml \
            --model-artifact-s3 s3://<PATH_TO_SAVE_TRAINED_MODEL> \
            --graph-name ogbn-mag \
            --task-type link_prediction \
            --num-layers 1 \
            --fanout 10 \
            --hidden-size 128 \
            --backend gloo \
            --batch-size 128

The above command will create a Docker Compose file named ``docker-compose-<task-type>-<num-instances>-train.yaml``, which we can then use to launch the job.

As our Docker Compose will use a Docker network, named ``gsf-network``, for inter-container communications, users need to run the following command to create the network before luanch Docker Compose.

.. code-block:: bash

    docker network create "gsf-network"

Then, use the following command to run the Link Prediction training on OGB-MAG graph.

.. code-block:: bash

    docker compose -f docker-compose-link_prediction-3-train.yaml up

Running the above command will launch 3 instances of the image, configured with the command and env vars that emulate a SageMaker execution environment and run the ``sagemaker_train.py`` script.

.. Note:: The containers actually interact with S3, so the provided AWS assess key, security access key, and session token should be valid for access S3 bucket.

Run GraphStorm on Docker Compose for Inference
```````````````````````````````````````````````
The ``generate_sagemaker_docker_compose.py`` can build Compose file for the inference task with the same arguments as for training, and in addition, but add a new argument, ``--inference``. The below command create the Compose file for the Link Prediction inference on OGB-MAG graph.

.. code-block:: bash

    python3 generate_sagemaker_docker_compose.py \
            --aws-access-key <<AWS_ACCESS_KEY>> \
            --aws-secret-access-key <AWS_SECRET_ACCESS_KEY> \
            --aws-session-token <AWS_SESSION_TOKEN> \
            --num-instances 3 \
            --image <GRAPHSTORM_DOCKER_IMAGE> \
            --graph-data-s3 s3://<PATH_TO_DATA>/ogbn_mag_lp_3p \
            --yaml-s3 s3://<PATH_TO_TRAINING_CONFIG>/map_lp.yaml \
            --model-artifact-s3 s3://<PATH_TO_SAVE_TRAINED_MODEL> \
            --graph-name ogbn-mag \
            --task-type link_prediction \
            --num-layers 1 \
            --fanout 10 \
            --hidden-size 128 \
            --backend gloo \
            --batch-size 128 \
            --inference

The command will create a Docker compose file named ``docker-compose-<task-type>-<num-instances>-infer.yaml``. And then, we can use the same command to spin up the inference job.

.. code-block:: bash

    docker compose -f docker-compose-link_prediction-3-infer.yaml up

Clean Up
``````````````````
To save computing resources, users can run the below command to clean up the Docker Compose environment.

.. code-block:: bash

    docker compose -f docker-compose-file down
