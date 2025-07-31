.. _real-time-inference-on-sagemaker:

Real-time Inference on Amazon SageMaker
----------------------------------------
GraphStorm CLIs for model inference on :ref:`signle machine <single-machine-training-inference>`,
:ref:`distributed clusters <distributed-cluster>`, and :ref:`Amazon SageMaker <distributed-sagemaker>`
are designed to handle enterprise-level data, which could take minutes to hours for predicting a large
number of target nodes/edges or generating embeddings for all nodes.

In certain cases when you want to predict a few targets only and hope to obtain results in short time,
e.g., less than one second, you will need a 7*24 running server to host trained model and response to
inference requests in real time. GraphStorm provides a CLI to deploy a trained model as a SageMaker
real-time inference endpoint. To invoke this endpoint, you will need to extract a subgraph around a few
target nodes/edges and convert the subgraph and associated features into a JSON object as payload of
invoke requests.

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

- **--image-uri** (Required): the URI of your GraphStorm real-time inference Docker image you built and
  pushed in the previous :ref:`Setup  GraphStorm Real-time Inference Docker Image <build_rt_inference_docker>` step.
- **--region** (Required): the AWS region to deploy endpoint. This region should be **same** as the ECR
  where your Docker image is stored.
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
- **--restore-model-path** (Required): the path where GraphStorm model parameters are saved.
- **--model-yaml-config-file** (Required): the path where updated model configuration YAML file is saved.
- **--graph-json-config-file** (Required): the path where updated graph construction configuration JSON file
  is saved.
- **--upload-tarfile-s3** (Required): the S3 location for uploading the packed and compressed model artifacts
  tar file.
- **--infer-task-type** (Required): the name of real-time inference task. Options include ``node_classification``
  and ``node_regression``.
- **--model-name** (Required): the name of model. This name will be used to define name of SageMaker Model,
  EndpointConfig, and Endpoint by appending datetime to this model name. The name should follow a regular
  expression pattern: ``^[a-zA-Z0-9]([\-a-zA-Z0-9]*[a-zA-Z0-9])$``. (Default: ``GSF-Model4Realtime``)

Outputs of the CLI include the deployed endpoint name based on the value for ``--model-name``, e.g.,
``GSF-Model4Realtime-2025-06-04-23-47-11``, to be used in the invoke step.

Invoke Real-time Inference Endpoint
.....................................
For real-time inference, you will need to extract a subgraph around the target nodes/edges from a large
graph, and use the subgraph as input of model, which is exactly how models are trained. Because time is
critical for real-time infernce, it is recommened to use OLTP graph database, e.g., Amazon Neptune Database,
as data source for subgraph extraction. 

Once the subgraph is extracted, you will need to prepare it as the payload of different APIs of `Invoke 
models for real-time inference
<https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-test-endpoints.html#realtime-endpoints-test-endpoints-api>`_.
GraphStorm defines a specification of the payload contents for your reference.

Payload Content specification
******************************

The payload should be a JSON object in the format explained below. In the highest level, the JSON object
contains four fields: ``version``, ``gml_task``, and ``graph``.

``version`` (**Required**)
>>>>>>>>>>>>>>>>>>>>>>>>>>>

This field is used to identify the version of a specification, helping to avoid compatibility issues. This
current (and expected) version is ``gs-realtime-v0.1``.

``gml_task`` (**Required**)
>>>>>>>>>>>>>>>>>>>>>>>>>>>

This field indicates what graph machine learning task this payload is for. Current built-in endpoint file
support two options: 

* ``node_classification``
* ``node_regression``

``graph`` (**Required**)
>>>>>>>>>>>>>>>>>>>>>>>>>

This ``graph`` field is similar to :ref:`graph construction JSON specification <_gconstruction-json>`. It
contains three types of sub-fields, i.e., ``nodes``, ``edges``, and ``targets``.

A ``nodes`` filed contains a list of ``node`` fileds. A ``node`` includes the raw input data values
of a node in the subgraph. It has the following required attributes.

* ``node_type``: string, the raw node type name in a graph. It should be same as these ``node_type`` defined in
  gconstruct JSON files or the ``type`` values of ``nodes`` defined in  in gsprocessing JSON files.
* ``node_id``: the raw node ID.
* ``features``: a dictionary, whose key is a feature name, and its value is the value of the feature.
  feaure names should be same as these ``feature_name`` defined in gconstruct JSON files, or these ``name``
  values of ``features`` fields defined defined in gsprocessing JSON files.

An ``edges`` field contains a list of ``edge`` fields. An ``edge`` includes the raw input data values of an
edge in the subgraph. It has the following required attributes.

* ``edge_type``: list, the raw edge type name in the format of a list with three elements, which indicate
  source node type, edge type, and destination edge type. It should be same as these ``relation`` fileds defined
  in gconstruct JSON files or these ``type`` values of ``source``, ``relation``, and ``dest`` fileds defined in
  gsprocessing JSON files.
* ``src_node_id``: user defined node ID for the source node.
* ``dest_node_id``: user defined node ID for the destination node.
* ``features``: a dictionary, whose key is a feature name, and its key is value of the feature. 
  feaure names should be same as these ``feature_name`` defined in gconstruct JSON files, or these ``name``
  values of ``features`` fields defined defined in gsprocessing JSON files.

A ``targets`` filed contains a list of target ``node`` or ``edge`` fileds depending on the value of ``gml_task``
field. These ``node`` or ``edge`` fileds is same as ``node`` and ``edge`` above, but the features field is not
required. And they should be in the ``nodes`` or ``edges`` list.

An example JSON file is like:

.. code:: yaml

    {
        "version": "gs-realtime-v0.1",
        "gml_task": "node_classification",
        "graph": {
            "nodes": [
                {
                    "node_type": "author",
                    "features": {
                        "feat": [
                            0.011269339360296726,
                            ......
                        ]
                    },
                    "node_id": "a4444"
                },
                {
                    "node_type": "author",
                    "features": {
                        "feat": [
                            -0.0032965524587780237,
                            .....
                        ]
                    },
                    "node_id": "s39"
                }
            ],
            "edges": [
                {
                    "edge_type": [
                        "author",
                        "writing",
                        "paper"
                    ],
                    "features": {},
                    "src_node_id": "p4463",
                    "dest_node_id": "p4463"
                },
                ......
            ]
        },
        "targets": [
            {
                "node_type": "paper",
                "node_id": "p4463"
            },
            or 
            {
                "edge_type": [
                        "paper",
                        "citing",
                        "paper"
                    ]
                "src_node_id": "p3551",
                "dest_node_id": "p3551"
            }
        ]
    }

Response from Endpoint
***********************
