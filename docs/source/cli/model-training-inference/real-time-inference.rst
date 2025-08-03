.. _real-time-inference-on-sagemaker:

Real-time Inference on Amazon SageMaker
----------------------------------------
GraphStorm CLIs for model inference on :ref:`signle machine <single-machine-training-inference>`,
:ref:`distributed clusters <distributed-cluster>`, and :ref:`Amazon SageMaker <distributed-sagemaker>`
are designed to tackle large dataset, which could take minutes to hours for predicting a large
number of target nodes/edges or generating embeddings for all nodes. In certain cases when you want to
predict a few targets only and expect to get results immediately, e.g., with in one second, you will need
a 7*24 running server to host trained model and response to inference requests in real time.

Since version 0.5, GraphStorm offers CLIs to deploy a trained model as a SageMaker real-time inference
endpoint. To invoke this endpoint, you will need to extract a subgraph around a few target nodes/edges,
convert it and associated features into a JSON object as payloads of requests. Below sections provide details
of how to deloy an endpoint, and how to invoke it for real-time infernce.

Prerequisites
..............
In order to use GraphStorm on Amazon SageMaker, users need to have an AWS account and access to the following AWS services.

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
Similarly to :ref:`GraphStorm model training and inference on SageMaker <distributed-sagemaker>`, you will
need to build a GraphStorm Docker image specifically for real-time inference. In addition, your executing role should
have full ECR access to be able to pull from ECR to build the image, create an ECR repository if it
doesn't exist, and push the real-time inference image to the repository. See the `official ECR docs
<https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-push-iam.html>`_ for details.

In short you can run the following:

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git
    cd graphstorm/

    # Build the GraphStorm real-time inference Docker image to be used on CPUs
    bash docker/build_graphstorm_image.sh --environment sagemaker-endpoint --device cpu

    # Will push an image to '123456789012.dkr.ecr.us-east-1.amazonaws.com/graphstorm:sagemaker-endpoint-cpu'
    bash docker/push_graphstorm_image.sh --environment sagemaker-endpoint --device cpu --region "us-east-1" --account "123456789012"

Replace the ``123456789012`` with your own AWS account ID. For more build and push options, see 
``bash docker/build_graphstorm_image.sh --help`` and ``bash docker/push_graphstorm_image.sh --help``.

.. note::

    CPU instances are more cost-effective and have similar inference latency as GPU instances. Therefore, it is
    recommended to use CPU instances for real-time inference.

Deploy a SageMaker Real-time Inference endpoint
................................................
To deploy a SageMaker real-time inference endpoint, you will need three model artifacts that were generated
during  graph construction (GConstruct/GSProcessing) and model training.

- The saved model path that contains the ``model.bin`` file. This path has the same purpose as the
  ``--restore-model-path`` used during offline inference CLIs in which GraphStorm looks for the ``model.bin``
  file, and uses it to restore the trained model weights.
- The updated graph construciton JSON file, ``data_transform_new.json``. This JSON file is one of the outputs of
  graph construction pipeline. It contains updated information about feature transformations and feature
  dimensions. If using the :ref:`Single Machine Graph Construction <single-machine-gconstruction>`(GConstruct), the
  file is saved at the path specified by the ``--output-dir`` argument. For :ref:`Distributed Graph Construction
  <distributed-gconstruction>`(GSProcessing), the file is saved at the path specified by either ``--output-data-s3``
  or ``--output-dir`` argument.
- The updated model training configuration YAML file, ``GRAPHSTORM_RUNTIME_UPDATED_TRAINING_CONFIG.yaml``. This
  YAML file is one of the outputs of model training. It contains the updated configurations of a model by
  replacing values of configuration YAML file with values given in the training CLI arguments. If set
  ``--save-model-path`` or ``--model-artifact-s3`` configuration in model training, this updated YAML file will
  be saved to the location specified.

.. note:: 

    Starting with v0.5, GraphStorm will save both updated JSON and YAML files into the same location as trained model
    automatically, if the ``--save-model-path`` or ``--model-artifact-s3``  configuration is set.

GraphStorm provides a helper script to package these model artifacts as a tar file and upload it to an S3 bucket, and then
deploy a SageMaker endpoint APIs with the inference Docker image previously built to deploy endpoint(s).

In short you can run the following:

.. code-block:: bash

    # assume graphstorm source code has been cloned to the current folder
    cd graphstorm/sagemaker/launch
    python launch_realtime_endpoint.py \
            --image-uri <account_id>.dkr.ecr.<region>.amazonaws.com/graphstorm:sagemaker-endpoint-cpu \
            --role arn:aws:iam::<account_id>:role/<your_role> \
            --region <region> \
            --restore-model-path <restore-model-path>/<epoch-XX-iter-XX> \
            --model-yaml-config-file <restore-model-path>/GRAPHSTORM_RUNTIME_UPDATED_TRAINING_CONFIG.yaml \
            --graph-json-config-file <restore-model-path>/data_transform_new.json \
            --infer-task-type node_classification \
            --upload-tarfile-s3 s3://<a-bucket> \
            --model-name <model-name>

Arguments of the launch endpoint script include:

- **-\-image-uri** (Required): the URI of your GraphStorm real-time inference Docker image built and
  pushed in the previous :ref:`Setup  GraphStorm Real-time Inference Docker Image <build_rt_inference_docker>` step.
- **-\-region** (Required): the AWS region to deploy endpoint. This region should be **same** as the ECR
  where your Docker image is stored.
- **-\-role** (Required): the role ARN that has SageMaker execution role. Please refer to the
  `SageMaker AI document <https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deploy-models.html#deploy-prereqs>`_
  section for details.
- **-\-instance-type**: the instance types to be used for endpoints. (Default: ``ml.c6i.xlarge``)
- **-\-instance-count**: the number of endpoints to be deployed. (Default: 1)
- **-\-custom-production-variant**: dictionary string that includes custom configurations of the SageMaker
  ProductionVariant. For details, please refer to `ProductionVariant Documentation
  <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html>`_.
- **-\-async-execution**: the mode of endpoint creation. Set ``True`` to deploy endpoint asynchronously,
  or ``False`` to wait for creation completed. (Default: ``True``)
- **-\-restore-model-path** (Required): the path where the ``model.bin`` file is saved.
- **-\-model-yaml-config-file** (Required): the path where the updated model configuration YAML file is saved.
- **-\-graph-json-config-file** (Required): the path where the updated graph construction configuration JSON file
  is saved.
- **-\-upload-tarfile-s3** (Required): the S3 location for uploading the packed and compressed model artifacts
  tar file.
- **-\-infer-task-type** (Required): the name of real-time inference task. Options include ``node_classification``
  and ``node_regression``.
- **-\-model-name**: the name of model. This name will be used to define names of SageMaker Model,
  EndpointConfig, and Endpoint by appending datetime to this model name. The name should follow a regular
  expression pattern: ``^[a-zA-Z0-9]([\-a-zA-Z0-9]*[a-zA-Z0-9])$``. (Default: ``GSF-Model4Realtime``)

This command will log out the deployed endpoint name based on the value for ``--model-name``, e.g.,
``GSF-Model4Realtime-Endpoint-2025-06-04-23-47-11``, to be used in the invoke step. The same endpoint name
can also be found from Amazon SageMaker AI Web console under the "Inference -> Endpoints" menu.

Invoke Real-time Inference Endpoints
.....................................
For real-time inference, you will need to extract a subgraph around the target nodes/edges from a large
graph, and use the subgraph as input of model, which is similar to how models are trained. Because time is
critical for real-time infernce, it is recommened to use OLTP graph database, e.g., Amazon Neptune Database,
as data source for subgraph extraction. 

Once the subgraph is extracted, you will need to prepare it as the payload of different APIs for `invoke 
models for real-time inference
<https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-test-endpoints.html#realtime-endpoints-test-endpoints-api>`_.
GraphStorm defines a specification of the payload contents.

.. _reat-time-payload-spec:

Payload content specification
******************************
The payload should be a JSON object in the format explained below. In the highest level, the JSON object
contains three fields: ``version``, ``gml_task``, and ``graph``.

``version`` (**Required**)
>>>>>>>>>>>>>>>>>>>>>>>>>>>
This field is used to identify the version of a specification, helping to avoid compatibility issues of different
versions. The current version is ``gs-realtime-v0.1``.

``gml_task`` (**Required**)
>>>>>>>>>>>>>>>>>>>>>>>>>>>
This field indicates what graph machine learning task this payload is for. Current specification supports two
options: 

* ``node_classification``
* ``node_regression``

``graph`` (**Required**)
>>>>>>>>>>>>>>>>>>>>>>>>>

This ``graph`` specifies the workload.
<gsprocessing_input_configuration>`. It contains three sub-fields, i.e., ``nodes``, ``edges``, and ``targets``.

A ``nodes`` field contains a list of ``node`` fileds. A ``node`` includes the raw input data values
of a node in the subgraph. It has the following required attributes.

* ``node_type``: string, the raw node type name in a graph. It should be same as these ``node_type`` defined in
  :ref:`gconstruct JSON specification <gconstruction-json>` or the ``type`` values of ``nodes`` defined in 
  :ref:`gsprocessing JSON specification <gsprocessing_input_configuration>`.
* ``node_id``: the raw node identifier.
* ``features``: a dictionary, whose key is a feature name, and its value is the value of features.
  feaure names should be same as the ``feature_name`` defined in :ref:`gconstruct JSON specification
  <gconstruction-json>`, or these ``name`` values of ``features`` fields defined in
  :ref:`gsprocessing JSON specification <gsprocessing_input_configuration>`.

An ``edges`` field contains a list of ``edge`` fields. An ``edge`` includes the raw input data values of an
edge in the subgraph. It has the following required attributes.

* ``edge_type``: list, the raw edge type name in the format of a list with three elements, which indicate
  source node type, edge type, and destination edge type. It should be same as the ``relation`` fileds defined
  in :ref:`gconstruct JSON specification <gconstruction-json>` or the ``type`` values of ``source``
  ``relation``, and ``dest`` fileds defined in :ref:`gsprocessing JSON specification <gsprocessing_input_configuration>`.
* ``src_node_id``: user defined node identifier for the source node.
* ``dest_node_id``: user defined node identifier for the destination node.
* ``features``: a dictionary, whose key is a feature name, and its key is value of the feature. 
  feaure names should be same as these ``feature_name`` defined in :ref:`gconstruct JSON specification
  <gconstruction-json>`, or these ``name`` values of ``features`` fields defined in
  :ref:`gsprocessing JSON specification <gsprocessing_input_configuration>`.

A ``targets`` field contains a list of target ``node`` or ``edge`` fileds depending on the value of ``gml_task``
These ``node`` or ``edge`` fileds is same as ``node`` and ``edge`` above, but the features field is not
required. And they should be in the ``nodes`` or ``edges`` list of a ``graph``.

An example payload JSON object is like the following:

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

Invoke endpoints
****************
There are multiple ways to invoke a Sagemaker real-time inference endpoint as documented in
`SageMaker Developer Guide <https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-test-endpoints.html#realtime-endpoints-test-endpoints-api>`_.

Here is an example of how you can read a payload from a JSON file and use the boto3 APIs to
invoke an endpoint.

.. code-block:: python

    import boto3
    import json

    # Create a SageMaker client object\n",
    sagemaker = boto3.client('sagemaker')
    # Create a SageMaker runtime client object using your IAM role ARN\n",
    runtime = boto3.client('sagemaker-runtime',
                           aws_access_key_id='your access key string',
                           aws_secret_access_key='your secret key string',
                           region_name='asw region' # e.g., us-east-1
    endpoint_name='your endpoint name'              # e.g., GraphStorm-Endpoint-2025-07-11-21-44-36
    # load payload from a JSON file
    with open('subg.json', 'r') as f:
         payload = json.load(f)
    content_type = 'application/json'

    # invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(payload),
        ContentType=content_type,
        )
    # Decodes and prints the response body
    print(response['Body'].read().decode('utf-8'))

The response format
********************
As shown in the previous invoke example, the response from GraphStorm's real-time inference endpoint will include
a JSON object in the ``Body`` field of the SageMaker API response. This JSON object contains five fields:

``status_code``
>>>>>>>>>>>>>>>>

The JSON object always includes a ``status_code`` field, which indicates the outcome status with an integer value,
including:

- ``200``: request processed successfully.
- ``400``: the request payload has JSON format errors.
- ``401``: the request payload missed certain fileds, required by :ref:`Payload specification <reat-time-payload-spec>`.
- ``402``: the request payload missed values on certain fileds.
- ``403``: ``node_type`` of nodes in the ``target`` field does not exist in the ``graph`` field.
- ``404``: values of the ``node_id`` fileds of nodes in the ``target`` field do not exist in the ``graph`` field.
- ``411``: errors occurred when converting the request payload into DGL graph format for inference.
- ``421``: the task in ``gml_task`` does not match the task that the deployed model is for.
- ``500``: internal server errors.

``request_uid``
>>>>>>>>>>>>>>>>

The JSON object always includes a ``request_uid`` field, which serves as a unique identifier for the request payload.
This identifier is logged on the endpoint side and returned to invokers, facilitating error debugging.

``message``
>>>>>>>>>>>>

The JSON object always include a ``message`` field, which provide additional information when the ``status_code`` is 200.

``error``
>>>>>>>>>>>>
The JSON object always include an ``error`` field, which provide detailed explanations when the ```status_code`` is not 200.

``data``
>>>>>>>>>
When the ``status_code`` is 200, the JSON object includes a populated ``data`` field. Otherwise, the data field is empty.

A ``200`` status response includes a JSON object containing inference results, with a single field called ``results``.
The values of ``results`` is a list that includes the inference values for all nodes specified in the payload's
``target`` field.

In addtion to the ``node_type`` and ``node_id`` fields, which match those in the payload ``target`` field, each result
in the list include a ``prediction`` field. This field contains the inference results for each node or edge. For
classification tasks, the value of ``prediction`` is a list of logits that can be used with classification method such
as `argmax`. For regression tasks, the value of ``prediction`` is a list with a single element, which represents the
regression result.

An example of a successful inference response:

.. code:: yaml

    {
        "status_code": 200,
        "request_uid": "569d90892909c2f8",
        "message": "Request processed successfully.",
        "error": "",
        "data": {
            "results": [
                {
                    "node_type": "paper",
                    "node_id": "p9604",
                    "prediction": [
                        0.03836942836642265,
                        0.06707385182380676,
                        0.11153795570135117,
                        0.027591131627559662,
                        0.03496604412794113,
                        0.11081098765134811,
                        0.005487487651407719,
                        0.027667740359902382,
                        0.11663214862346649,
                        0.11842530965805054,
                        0.020509174093604088,
                        0.031869057565927505,
                        0.27694952487945557,
                        0.012110156007111073
                    ]
                },
                {
                    "node_type": "paper",
                    "node_id": "p8946",
                    "prediction": [
                        0.03848873823881149,
                        0.06991259753704071,
                        0.057228244841098785,
                        0.02898392826318741,
                        0.046037621796131134,
                        0.09567245841026306,
                        0.008081010542809963,
                        0.02855496294796467,
                        0.2774551510810852,
                        0.07382062822580338,
                        0.03699302300810814,
                        0.047642651945352554,
                        0.1794610172510147,
                        0.011668065562844276
                    ]
                }
            ]
        }
    }

An example of an error response:

.. code:: yaml

    {
        "status_code": 401,
        "request_uid": "d3f2eaea2c2c7c76",
        "message": "",
        "error": "Missing Required Field: The input payload missed the 'targets' field. Please refer to the GraphStorm realtime inference documentation for required fields.",
        "data": {}
    }
