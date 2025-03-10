.. _graphstorm-sagemaker-pipeline-ref:

Using GraphStorm with SageMaker Pipelines
=========================================

GraphStorm provides integration with Amazon SageMaker Pipelines to automate and orchestrate graph machine learning workflows at scale.
This guide shows you how to use the provided tools to create, configure, and execute SageMaker pipelines for graph construction, training, and inference.

Introduction
------------

SageMaker Pipelines enable you to create automated MLOps workflows for your GraphStorm applications. Using these workflows you can:

* Automate the end-to-end process of preparing graph data, training models, and running inference
* Ensure reproducibility of your graph machine learning experiments
* Scale your workflows efficiently using SageMaker's managed infrastructure
* Track and version your pipeline executions

Pre-requisites
--------------

Before starting with GraphStorm Pipelines on SageMaker, you'll need:

* An execution environment with Python 3.8 or later
* An AWS account with appropriate permissions. See the official
  `SageMaker documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-access.html>`_
  for detailed information about required permissions.
* Basic familiarity with Amazon SageMaker and
  `SageMaker Pipelines <https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html>`_.
* Understanding of graph neural networks and the `GraphStorm framework <https://graphstorm.readthedocs.io/en/latest/index.html>`_.

Setting Up Your Environment
---------------------------

To work with GraphStorm SageMaker pipelines, you'll need the GraphStorm source code
and a Python environment with the SageMaker SDK and AWS SDK (boto3) installed.

1. First, clone the GraphStorm repository and navigate to the pipeline directory:

.. code-block:: bash

   git clone https://github.com/awslabs/graphstorm.git
   cd graphstorm/sagemaker/pipeline

2. Install the required Python packages:

.. code-block:: bash

   pip install sagemaker boto3

Creating Your First Pipeline
--------------------------

GraphStorm provides Python scripts to help you create and manage SageMaker pipelines. The main tools are:

* ``create_sm_pipeline.py``: Creates or updates pipeline definitions
* ``pipeline_parameters.py``: Manages pipeline configuration
* ``execute_sm_pipeline.py``: Runs pipelines

Here's an example of how to create a basic pipeline that includes graph construction, training, and inference:

.. code-block:: bash

    python create_sm_pipeline.py \
        --graph-construction-config-filename my_gconstruct_config.json \
        --graph-name my-graph \
        --graphstorm-pytorch-cpu-image-url 123456789012.dkr.ecr.us-west-2.amazonaws.com/graphstorm:sagemaker-cpu \
        --input-data-s3 s3://input-bucket/data \
        --instance-count 2 \
        --jobs-to-run gconstruct train inference \
        --output-prefix s3://output-bucket/results \
        --pipeline-name my-graphstorm-pipeline \
        --region us-west-2 \
        --role arn:aws:iam::123456789012:role/SageMakerExecutionRole \
        --train-inference-task node_classification \
        --train-yaml-s3 s3://config-bucket/train.yaml

This command sets up a pipeline with three main stages:

1. Graph construction using the configuration in ``my_gconstruct_config.json``
2. Model training using the settings in ``train.yaml``
3. Inference using the trained model

The pipeline will use 2 instances (specified by ``--instance-count``) for distributed training and inference.

You'll need to provide:

* A SageMaker execution role (``--role``) with appropriate permissions
* A GraphStorm Docker image (``--graphstorm-pytorch-cpu-image-url``) for running the tasks
* S3 locations for your input data and where to store results

Running Pipeline Executions
-------------------------

Once you've created a pipeline, you can execute it using the ``execute_sm_pipeline.py`` script:

.. code-block:: bash

    python execute_sm_pipeline.py \
        --pipeline-name my-graphstorm-pipeline \
        --region us-west-2

You can override default parameters during execution to customize the run:

.. code-block:: bash

    python execute_sm_pipeline.py \
        --pipeline-name my-graphstorm-pipeline \
        --region us-west-2 \
        --instance-count 4 \
        --gpu-instance-type ml.g4dn.12xlarge

Pipeline Components
-----------------

A GraphStorm SageMaker pipeline can include several components that you can combine based on your needs.
We list those here, with the step name that you can provide in ``--jobs-to-run`` in parentheses.

1. **Single-instance Graph Construction** (``gconstruct``):
   Single-instance graph construction for small graphs.

2. **Distributed Graph pre-processing** (``gsprocessing``):
   PySpark-based distributed data preparation for large graphs.

3. **Distributed Graph Partitioning** (``dist_part``):
   Multi-instance graph partitioning for distributed training.

4. **GraphBolt Conversion** (``gb_convert``):
   Converts partitioned data to GraphBolt format for improved training/inference efficiency..

5. **Training** (``train``):
   Trains your graph neural network model.

6. **Inference** (``inference``):
   Runs predictions using your trained model.

Configuration Options
---------------------

This section provides a comprehensive list of all available configuration options for creating and executing GraphStorm SageMaker pipelines.

AWS Configuration
^^^^^^^^^^^^^^^^^

* ``--execution-role``: SageMaker execution IAM role ARN. (Required)
* ``--region``: AWS region. (Required)
* ``--graphstorm-pytorch-cpu-image-uri``: GraphStorm GConstruct/dist_part/train/inference CPU ECR image URI. (Required)
* ``--graphstorm-pytorch-gpu-image-uri``: GraphStorm GConstruct/dist_part/train/inference GPU ECR image URI.
* ``--gsprocessing-pyspark-image-uri``: GSProcessing SageMaker PySpark ECR image URI. (Required if running a ``gsprocessing`` job.)

Instance Configuration
^^^^^^^^^^^^^^^^^^^^^^

* ``--instance-count`` / ``--num-parts``: Number of worker instances/partitions for partition, training, inference. (Required)
* ``--cpu-instance-type``: CPU instance type. (Default: ml.m5.4xlarge)
* ``--gpu-instance-type``: GPU instance type. (Default: ml.g5.4xlarge)
* ``--train-on-cpu``: Run training and inference on CPU instances instead of GPU. (Flag)
* ``--graph-construction-instance-type``: Instance type for graph construction.
* ``--gsprocessing-instance-count``: Number of GSProcessing instances (PySpark cluster size, default is equal to ``--instance-count``).
* ``--volume-size-gb``: Additional volume size for SageMaker instances in GB. (Default: 100)

Task Configuration
^^^^^^^^^^^^^^^^^^

* ``--graph-name``: Name of the graph. (Required)
* ``--input-data-s3``: S3 path to the input graph data. (Required)
* ``--output-prefix-s3``: S3 prefix for the output data. (Required)
* ``--pipeline-name``: Name for the pipeline.
* ``--base-job-name``: Base job name for SageMaker jobs. (Default: 'gs')
* ``--jobs-to-run``: Space-separated strings of jobs to run in the pipeline.
  Possible values are: ``gconstruct``, ``gsprocessing``, ``dist_part``, ``gb_convert``, ``train``, ``inference`` (Required).
* ``--log-level``: Logging level for the jobs. (Default: INFO)
* ``--step-cache-expiration``: Expiration time for the step cache. (Default: 30d)
* ``--update-pipeline``: Update an existing pipeline instead of creating a new one. (Flag)

Graph Construction Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``--graph-construction-config-filename``: Filename for the graph construction config.
* ``--graph-construction-args``: Additional parameters to be passed directly to the GConstruct/GSProcessing job.

Partition Configuration
^^^^^^^^^^^^^^^^^^^^^^^

* ``--partition-algorithm``: Partitioning algorithm to use. (Default: random)
* ``--partition-input-json``: Name for the JSON file that describes the input data for distributed partitioning. (Default: updated_row_counts_metadata.json)
* ``--partition-output-json``: Name for the output JSON file that describes the partitioned data generated by GConstruct or GSPartition.
  (Default: metadata.json for GSPartition,  use <graph_name>.json for ``gconstruct``.)

Training Configuration
^^^^^^^^^^^^^^^^^^^^^^

* ``--model-output-path``: S3 path for model output.
* ``--num-trainers``: Number of trainers (per-instance training processes) to use during training/inference. Set this equal to number of GPUs (Default: 4)
* ``--train-inference-task-type``: Task type for training and inference. (Required)
* ``--train-yaml-s3``: S3 path to the train YAML configuration file.
* ``--use-graphbolt``: Whether to use GraphBolt for GConstruct, training and inference. (Default: false)

Inference Configuration
^^^^^^^^^^^^^^^^^^^^^^^

* ``--inference-yaml-s3``: S3 path to inference YAML configuration file.
* ``--inference-model-snapshot``: Which model snapshot to choose to run inference with, e.g. ``epoch-9`` to use the model generated by the 10th (zero-indexed) epoch.
* ``--save-predictions``: Whether to save predictions to S3 during inference. (Flag)
* ``--save-embeddings``: Whether to save embeddings to S3 during inference. (Flag)

Script Paths
^^^^^^^^^^^^

* ``--dist-part-script``: Path to DistPartition SageMaker entry point script.
* ``--gb-convert-script``: Path to GraphBolt partition conversion script.
* ``--train-script``: Path to training SageMaker entry point script.
* ``--inference-script``: Path to inference SageMaker entry point script.
* ``--gconstruct-script``: Path to GConstruct SageMaker entry point script.
* ``--gsprocessing-script``: Path to GSProcessing SageMaker entry point script.

Using Configuration Options (Example)
---------------------------

When creating or executing a pipeline, you can use these options to customize your workflow. For example:

.. code-block:: bash

    python create_sm_pipeline.py \
        --graph-name my-large-graph \
        --input-data-s3 s3://my-bucket/input-data \
        --output-prefix-s3 s3://my-bucket/output \
        --instance-count 4 \
        --gpu-instance-type ml.g4dn.12xlarge \
        --jobs-to-run gsprocessing dist_part gb_convert train inference \
        --use-graphbolt true \
        --train-yaml-s3 s3://my-bucket/train-config.yaml \
        --inference-yaml-s3 s3://my-bucket/inference-config.yaml \
        --save-predictions \
        --save-embeddings

This example sets up a pipeline for a large graph, using distributed processing, GraphBolt conversion, GPU-based training and inference, and saving both predictions and embeddings.

Remember that not all options are required for every pipeline. The necessary options depend on your specific use case and the components you're including in your pipeline.

Advanced Usage
------------

Using GraphBolt for Better Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GraphBolt enabled faster training, see :ref:`using-graphbolt-ref`. To enable GraphBolt for your pipeline:

.. code-block:: bash

    python create_sm_pipeline.py \
        ... \
        --use-graphbolt true

For distributed processing with GraphBolt, you will need to include a ``gb_convert`` step after ``dist_part``:

.. code-block:: bash

    python create_sm_pipeline.py \
        ... \
        --jobs-to-run gsprocessing dist_part gb_convert train inference \
        --use-graphbolt true

For a complete example of running a GraphBolt-enabled pipeline see this `AWS ML blog post <https://aws.amazon.com/blogs/machine-learning/faster-distributed-graph-neural-network-training-with-graphstorm-v0-4/>`_.

Asynchronous and Local Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For non-blocking pipeline execution:

.. code-block:: bash

    python execute_sm_pipeline.py \
        --pipeline-name my-graphstorm-pipeline \
        --region us-west-2 \
        --async-execution

For local testing, where all pipeline steps are executed locally:

.. code-block:: bash

    python execute_sm_pipeline.py \
        --pipeline-name my-graphstorm-pipeline \
        --local-execution

.. note:: Local execution requires a GPU if using GPU instance types.

Troubleshooting
---------------

If you encounter issues:

* Check that all AWS permissions are correctly configured
* Review SageMaker execution logs for detailed error messages
* Verify S3 path accessibility
* Confirm instance type availability in your region

For more information, see:

* `SageMaker Pipelines Troubleshooting Guide <https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-troubleshooting.html>`_

For additional help, you can open an issue in the
`GraphStorm GitHub repository <https://github.com/awslabs/graphstorm/issues>`_.
