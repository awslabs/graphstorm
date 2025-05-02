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

Building and pushing a SageMaker uses the same scripts as for building a local image,
described in :ref:`GraphStorm Docker build instructions <_build_docker>`.

Your executing role should have full ECR access to be able to pull from ECR to build the image,
create an ECR repository if it doesn't exist, and push the GSProcessing image to the repository.
See the [official ECR docs](https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-push-iam.html)
for details.

In short you can run the following:

.. code-block:: bash

    cd graphstorm/
    bash docker/build_graphstorm_image.sh --environment sagemaker
    bash docker/push_graphstorm_image.sh --environment sagemaker --region "us-east-1" --account "123456789012"
    # Will push an image to '123456789012.dkr.ecr.us-east-1.amazonaws.com/graphstorm:sagemaker-gpu'

See ``bash docker/build_graphstorm_image.sh --help`` and ``bash docker/push_graphstorm_image.sh --help``
for more build and push options.

Run GraphStorm on SageMaker
----------------------------

To run GraphStorm with the Amazon SageMaker service, users should set up a local Python environment with the SageMaker library installed and GraphStorm's SageMaker helper scripts.

1. Use the below command to install SageMaker.

.. code-block:: bash

    pip install --upgrade sagemaker

2. Clone the GraphStorm repository using the following command

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git
    # Change to the GraphStorm sagemaker directory
    cd graphstorm

For the remainder of this guide we assume the starting working directory is the root
of the GraphStorm repository.

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

Launch embedding generation task
``````````````````````````````````
Users can use the following example command to launch a GraphStorm embedding generation job in the ``ogbn-mag`` data without generating predictions.

.. code:: bash

    python3 launch/launch_infer.py  \
            --image-url <AMAZON_ECR_IMAGE_URI> \
            --region <REGION> \
            --entry-point run/infer_entry.py \
            --role <ROLE_ARN> \
            --instance-count 3 \
            --graph-data-s3 s3://<PATH_TO_DATA>/ogbn_mag_lp_3p \
            --yaml-s3 s3://<PATH_TO_TRAINING_CONFIG>/mag_lp.yaml \
            --model-artifact-s3 s3://<PATH_TO_SAVE_TRAINED_MODEL>/ \
            --raw-node-mappings-s3 s3://<PATH_TO_DATA>/ogbn_mag_lp_3p/raw_id_mappings \
            --task-type compute_emb \
            --output-emb-s3 s3://<PATH_TO_SAVE_GENERATED_NODE_EMBEDDING>/ \
            --graph-name ogbn-mag \
            --restore-model-layers embed,gnn


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

Launch hyper-parameter optimization task
````````````````````````````````````````

GraphStorm supports `automatic model tuning <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html>`_
with SageMaker AI,
which allows you to optimize the hyper-parameters
of your model with an easy-to-use interface.

The ``sagemaker/launch/launch_hyperparameter_tuning.py`` script can act as a thin
wrapper for SageMaker's `HyperParameterTuner <https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html>`_.

You define the hyper-parameters of interest by passing a filepath to a JSON file,
or a python dictionary as a string,
where the structure of the dictionary is the same as for SageMaker's
`Dynamic hyper-parameters <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-ranges.html#automatic-model-tuning-define-ranges-dynamic>`.
For example your JSON file can look like:

.. code:: python

    # Content of my_param_ranges.json
    {
        "ParameterRanges": {
            "CategoricalParameterRanges": [
                {
                    "Name": "model_encoder_type",
                    "Values": ["rgcn", "hgt"]
                }
            ],
            "ContinuousParameterRanges": [
                {
                    "Name": "lr",
                    "MinValue": "1e-5",
                    "MaxValue" : "1e-2",
                    "ScalingType": "Auto"
                }
            ],
            "IntegerParameterRanges": [
                {
                    "Name": "hidden_size",
                    "MinValue": "64",
                    "MaxValue": "256",
                    "ScalingType": "Auto"
                }
            ]
        }
    }

Which you can then use to launch an HPO job:

.. code:: bash

    # Example hyper-parameter ranges
    python launch/launch_hyperparameter_tuning.py \
        --hyperparameter-ranges my_param_ranges.json
        # Other launch parameters...

For continuous and integer parameters you can provide a ``ScalingType``
string that directly corresponds to one of SageMaker's
`scaling types <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-ranges.html#scaling-type>`_.
By default scaling type will be ``'Auto'``.

Use ``--metric-name`` to provide the name of a GraphStorm metric to use as a tuning objective,
e.g. ``"accuracy"``. See the entry for ``eval_metric`` in :ref:`Evaluation Metrics <eval_metrics>`
for a full list of supported metrics.

``--eval-mask`` defines which dataset to collect metrics from, and
can be either ``"test"`` or ``"val"`` to collect metrics from test or validation set,
respectively. Finally use ``--objective-type`` to set the type of the objective,
which can be either ``"Maximize"`` or ``"Minimize"``.
See the `SageMaker documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html>`_
for more details

Finally you can use ``--strategy`` to select the optimization strategy
from one of "Bayesian", "Random", "Hyperband", "Grid". See the
`SageMaker documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html>`_
for more details on each strategy.

To use the Hyperband strategy you should provide the ``--hb-max-epochs`` and ``--hb-min-epochs`` to the launch script
to determine the maximum and minimum resource allocation (in terms of number of epochs) per job. See the
`SageMaker HPO user guide <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html#automatic-tuning-hyperband>`_
and `Hyperband configuration docs <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_HyperbandStrategyConfig.html>`_
for details.

Example HPO call:

.. code:: bash

    python launch/launch_hyperparameter_tuning.py \
        --task-name my-gnn-hpo-job \
        --role arn:aws:iam::123456789012:role/SageMakerRole \
        --region us-west-2 \
        --image-url 123456789012.dkr.ecr.us-west-2.amazonaws.com/graphstorm:sagemaker-gpu \
        --graph-name my-graph \
        --task-type node_classification \
        --graph-data-s3 s3://my-bucket/graph-data/ \
        --yaml-s3 s3://my-bucket/train.yaml \
        --model-artifact-s3 s3://my-bucket/model-artifacts/ \
        --max-jobs 20 \
        --max-parallel-jobs 4 \
        --hyperparameter-ranges my_param_ranges.json \
        --metric-name "accuracy" \
        --eval-mask "val" \
        --objective-type "Maximize" \
        --strategy "Bayesian"

Passing additional arguments to the SageMaker Estimator
```````````````````````````````````````````````````````
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


Run GraphStorm SageMaker jobs locally
.....................................

You can use `SageMaker's local mode <https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode>`_
to test running your jobs locally before launching large-scale
jobs, to ensure your configuration or other changes are correct.

First, you need to ensure your SageMaker installation has the necessary dependencies.
SageMaker Local Mode requires
`Docker Compose <https://docs.docker.com/compose/install/#scenario-two-install-the-docker-compose-plugin-linux-only>`_
and a SageMaker Python SDK installation with ``local`` extras:

.. code:: bash

    pip install 'sagemaker[local]' --upgrade

second, ensure your local mode configuration includes a high shared memory size for the
launched local containers:

.. code:: yaml

    local:
        local_code: true # Using everything locally
        region_name: "us-west-2" # Name of the region
        container_config:
            shm_size: "32G" # Set this according to your available memory

This is necessary as GraphStorm uses shared memory for in-memory graph storage.

Set the environment variable ``USE_SHORT_LIVED_CREDENTIALS=1`` if running on EC2 and
you would like to use the session credentials instead of EC2 Metadata Service credentials:

.. code:: bash

    export USE_SHORT_LIVED_CREDENTIALS=1

finally, when launching your SageMaker job, use ``local`` as the instance type:

.. code:: bash

    python3 launch/launch_[infer|train|partition] \
       <other arguments> \
       --instance-type "local"

The above will launch the GraphStorm job by spinning up local
Docker containers, using Docker compose.

Legacy image building instructions
``````````````````````````````````

Since GraphStorm 0.4.0 we provide new build scripts to facilitate easier image building
and pushing to ECR. In this section we provide the instructions for the legacy scripts.
These scripts will be deprecated in version 0.5 and removed in a future version of GraphStorm.

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
