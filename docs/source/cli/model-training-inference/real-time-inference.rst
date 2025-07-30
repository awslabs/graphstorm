.. _real-time-inference-on-sagemaker:

Real-time Inference on Amazon SageMaker
----------------------------------------
GraphStorm CLIs for model inference on :ref:`signle machine <single-machine-training-inference>`,
:ref:`distributed clusters <distributed-cluster>`, and :ref:`Amazon SageMaker <distributed-sagemaker>`
are designed to handle enterprise-level data, which could take minutes to hours for predicting a large
number of target nodes/edges or generating embeddings for all nodes.

In certain cases when you want to predict a few targets only and hope to obtain results in short time, e.g., less than one second, you will need a 7*24 running server to host trained model and response to
inference request in real time. GraphStorm provides a CLI to deploy a trained model as a SageMaker
real-time inference endpoint. To invoke this endpoint, you will need to extract a subgraph around a few
target nodes/edges and convert the subgraph and associated features into a JSON object as payload of
an invoke request.

Prerequisites
..............
In order to use GraphStorm on Amazon SageMaker, users need to have AWS access to the following AWS services.

- **SageMaker service**. Please refer to `Amazon SageMaker service <https://aws.amazon.com/pm/sagemaker/>`_
  for how to get access to Amazon SageMaker.
- **Amazon ECR**. Please refer to `Amazon Elastic Container Registry service <https://aws.amazon.com/ecr/>`_
  for how to get access to Amazon ECR.
- **S3 service**. Please refer to `Amazon S3 service <https://aws.amazon.com/s3/>`_
  for how to get access to Amazon S3.
- **SageMaker Framework Containers**. Please follow `AWS Deep Learning Containers guideline <https://github.com/aws/deep-learning-containers>`_
  to get access to the image.
- **Amazon EC2** (optional). Please refer to `Amazon EC2 service <https://aws.amazon.com/ec2/>`_
  for how to get access to Amazon EC2.

.. _build_rt_inference_docker:

Setup GraphStorm Real-time Inference Docker Image
..................................................
Same as :ref:`GraphStorm model training and inference on SageMaker <distributed-sagemaker>`, you will
need to build a GraphStorm real-time inference Docker image. In addtion, your executing role should
have full ECR access to be able to pull from ECR to build the image, create an ECR repository if it
doesn't exist, and push the real-time inference image to the repository. See the `official ECR docs
<https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-push-iam.html>`_ for details.

In short you can run the following:

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git
    cd graphstorm/
    # Build the GraphStorm real-time inference Docker image using CPU
    bash docker/build_graphstorm_image.sh --environment sagemaker-endpoint --device cpu
    # Will push an image to '123456789012.dkr.ecr.us-east-1.amazonaws.com/graphstorm:sagemaker-endpoint-cpu'
    bash docker/push_graphstorm_image.sh --environment sagemaker-endpoint --device cpu --region "us-east-1" --account "123456789012"

Replace the ``123456789012`` with your own AWS account ID. For more build and push options, see 
``bash docker/build_graphstorm_image.sh --help`` and ``bash docker/push_graphstorm_image.sh --help``.

Deploy a SageMaker Real-time Inference endpoint
................................................
To deploy a SageMaker real-time inference endpoint, you will need three model artifacts that generated from
graph construciton and model training.

- Saved model path that contains the ``model.bin`` file. This path has the same purpose as the
  ``--restore-model-path`` used during offline inference CLIs in which GraphStorm looks for the ``model.bin``
  file to restore a model at endpoint.
- Updated graph construciton JSON file, ``data_transform_new.json``. This JSON file is one of the outputs of
  graph construction. It contains the updated information about feature transformation and feature
  dimensions. If using the :ref:`Single Machine Graph Construction <single-machine-gconstruction>` CLIs, the
  file is saved at the path specified by the ``--output-dir`` argument. For :ref:`Distributed Graph Construction
  <distributed-gconstruction>` CLIs, the file is saved at the path specified by either ``--output-data-s3``
  or ``--output-dir`` argument.
- Updated model training configuration YAML file, ``GRAPHSTORM_RUNTIME_UPDATED_TRAINING_CONFIG.yaml``. This
  YAML file is one of the outputs of model training. It contains the updated configurations of a model by
  updating the values of configuration YAML file with values given in CLIs arguments. If set
  ``--save-model-path`` or ``--model-artifact-s3`` configuration, this updated YAML file will be saved to
  the location specified.

.. note:: 

    Since v0.5, GraphStorm will save both updated JSON and YAML files into the same location as trained model
    automatically, if the ``--save-model-path`` or ``--model-artifact-s3``  configuration is set.

GraphStorm provides CLIs to package these model artifacts as a tar file and upload it to an S3 bucket, and then
invoke SageMaker endpoint APIs with the inference Docker image previousely built and the tar file to deploy
endpoint(s).

In short you can run the following:

.. code-block:: bash

    # assume graphstorm source code has been cloned to the current folder
    cd graphstorm/sagemaker/launch
    python launch_realtime_endpoint.py \
        --image-uri <account_id>.dkr.ecr.<region>.amazonaws.com/graphstorm:sagemaker-endpoint-cpu \
        --role arn:aws:iam::<account_id>:role/<your_role> \
        --region <region> \
        --restore-model-path <restore-model-path>/<epoch-XX> \
        --model-yaml-config-file /<path-to-yaml>/GRAPHSTORM_RUNTIME_UPDATED_TRAINING_CONFIG.yaml \
        --graph-json-config-file /<path-to-json>/data_transform_new.json \
        --infer-task-type node_classification \
        --upload-tarfile-s3 s3://<a-bucket> \
        --model-name <model-name>

Arguments of the launch CLI include:

- **--image-uri** (Required): the URI of your GraphStorm real-time inference Docker image you built and pushed in the
  previous :ref:`Setup  GraphStorm Real-time Inference Docker Image <build_rt_inference_docker>` step.
- **--region** (Required): the AWS region to deploy endpoint. This region should be **same** as the ECR where your Docker
  image is stored.
- **--role** (Required): the role ARN that has SageMaker execution role. Please refer to the
  `Configure <https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deploy-models.html#deploy-models-python>`_
  section for details.
- **--instance-type**: the instance types to be used for endpoints. (Default: ``ml.c6i.xlarge``)
- **--instance-count**: the number of endpoints to be deployed. (Default: 1)
- **--custom-production-variant**: dictionary string that includes custom configurations of the SageMaker
  ProductionVariant. For details, please refer to `ProductionVariant Documentation
  <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html>`_.
- **--async-execution**: the mode of endpoint creation. Set ``True`` to deploy endpoint asynchronously,
  or ``False`` to wait for creation completed. (Default: ``True``)
- **--restore-model-path**(Required): the path where GraphStorm model parameters were saved.
- **--model-yaml-config-file**(Required):

While the :ref:`Standalone Mode Quick Start <quick-start-standalone>` tutorial introduces some basic concepts, commands, and steps of using GprahStorm CLIs on a single machine, this user guide provides more detailed description of the usage of GraphStorm CLIs in a single machine. In addition, the majority of the descriptions in this guide can be directly applied to :ref:`model training and inference on distributed clusters <distributed-cluster>`.

GraphStorm can support graph machine learning (GML) model training and inference for common GML tasks, including **node classification**, **node regression**, **edge classification**, **edge regression**, and **link prediction**. Since the :ref:`multi-task learning <multi_task_learning>` feature released in v0.3 is in experimental stage, formal documentations about this feature will be released later when it is mature.

For each task, GraphStorm provide a dedicated CLI for model training and inference. These CLIs share the same command template and some configurations, while each CLI has its unique task-specific configurations. GraphStorm also has a task-agnostic CLI for users to run your customized models.

While the CLIs could be very simple as the template demonstrated, users can leverage a YAML file to set a variaty of GraphStorm configurations that could make full use of the rich functions and features provided by GraphStorm. The YAML file will be specified to the **-\-cf** argument. GraphStorm has a set of `example YAML files <https://github.com/awslabs/graphstorm/tree/main/training_scripts>`_ available for reference.

.. note:: 

    * Users can set CLI configurations either in CLI arguments or the configuration YAML file. But values set in CLI arguments will overwrite the values of the same configuration set in the YAML file.
    * This guide only explains a few commonly used configurations. For the detailed explanations of GraphStorm CLI configurations, please refer to the :ref:`Model Training and Inference Configurations <configurations-run>` section.

Task-agnostic CLI for model training and inference
...................................................
While task-specific CLIs allow users to quickly perform GML tasks supported by GraphStorm, users may build their own GNN models as described in the :ref:`Use Your Own Models <use-own-models>` tutorial. To put these customized models into GraphStorm model training and inference pipeline, users can use the task-agnostic CLI as shown in the examples below.


.. code-block:: bash

    # Model training
    python -m graphstorm.run.launch \
              --num-trainers 1 \
              --part-config data.json \
              customized_model.py --save-model-path model_path/ \
                                  customized_arguments 

    # Model inference
    python -m graphstorm.run.launch \
              --inference \
              --num-trainers 1 \
              --part-config data.json \
              customized_model.py --restore-model-path model_path/ \
                                  --save-prediction-path pred_path/ \
                                  customized_arguments

The task-agnostic CLI command (``launch``) has similar tempalte as the task-specific CLIs except that it takes the customized model, which is stored as a ``.py`` file, as an argument. And in case the customized model has its own arguments, they should be placed after the customized model python file.
