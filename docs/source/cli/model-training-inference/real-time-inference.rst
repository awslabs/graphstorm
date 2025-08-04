.. _real-time-inference-on-sagemaker:

========================================
Real-time Inference on Amazon SageMaker
========================================

GraphStorm CLIs for model inference on :ref:`single machine <single-machine-training-inference>`,
:ref:`distributed clusters <distributed-cluster>`, and :ref:`Amazon SageMaker <distributed-sagemaker>`
are designed to handle large datasets, which could take minutes to hours for predicting a large number of target
nodes/edges or generating embeddings for all nodes. This is typically referred to as offline inference, where the
model processes a large batch of data at once, and the results are not needed immediately.

However, in certain use cases such as recommendation systems, social network analysis, and fraud detection, there
is a need for real-time predictions. For instance, you may only want to predict a few targets and expect to get
results immediately, say within one second. In these scenarios, you will need a 7*24 running server to host the
trained model and respond to inference requests in real time. This is known as online inference, where the model
is constantly available to make predictions on new data as it comes in, ensuring immediate responses for
time-sensitive applications.

Since version 0.5, GraphStorm offers new features that can deploy a trained model as a SageMaker real-time
inference endpoint. Below sections provide details of how to deloy an endpoint, and how to invoke it.

Prerequisites
--------------
In order to use GraphStorm real-time inference on Amazon SageMaker, users need to have an AWS account and access
to the following AWS services.

- **SageMaker service**. Needed to deploy endpoint, and optionally train models. Please refer to `Amazon SageMaker service <https://aws.amazon.com/pm/sagemaker/>`_
  for how to get access to Amazon SageMaker.
- **Amazon ECR**. Needed to store GraphStorm Sagemaker Docker images. Please refer to `Amazon Elastic Container Registry service <https://aws.amazon.com/ecr/>`_
  for how to get access to Amazon ECR.
- **S3 service**. Needed for input and output for SageMaker. Please refer to `Amazon S3 service <https://aws.amazon.com/s3/>`_
  for how to get access to Amazon S3.

.. _build_rt_inference_docker:

Setup GraphStorm Real-time Inference Docker Image
-------------------------------------------------

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

    When comparing CPU instances to GPU instances for real-time inference, CPU instances prove more
    cost-effective while maintaining similar inference latency to GPU instances. While CPU
    instances are generally recommended for real-time inference, users are encouraged to run their own
    benchmarks and cost analysis to make the final decision between CPU and GPU instances based on their
    particular workload needs.

Deploy a SageMaker Real-time Inference endpoint
------------------------------------------------

To deploy a SageMaker real-time inference endpoint, you will need three model artifacts that were generated
during  graph construction (GConstruct/GSProcessing) and model training.

- The saved model path that contains the ``model.bin`` file. This path has the same purpose as the
  ``--restore-model-path`` used during offline inference CLIs in which GraphStorm looks for the ``model.bin``
  file, and uses it to restore the trained model weights.
- The updated graph construciton JSON file, ``data_transform_new.json``. This JSON file is one of the outputs of
  graph construction pipeline. It contains updated information about feature transformations and feature
  dimensions. If using the :ref:`Single Machine Graph Construction <single-machine-gconstruction>` (GConstruct), the
  file is saved at the path specified by the ``--output-dir`` argument. For :ref:`Distributed Graph Construction
  <distributed-gconstruction>` (GSProcessing), the file is saved at the path specified by either ``--output-data-s3``
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
use SageMaker APIs with the inference Docker image previously built to deploy endpoint(s).

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
            --infer-task-type <task_type> \
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
  or ``False`` to wait for deployment to finish before exiting. (Default: ``True``)
- **-\-restore-model-path** (Required): a local folder path where the ``model.bin`` file is saved.
- **-\-model-yaml-config-file** (Required): a local file path where the updated model configuration YAML file
  is saved.
- **-\-graph-json-config-file** (Required): a local file path where the updated graph construction configuration
  JSON file is saved.
- **-\-upload-tarfile-s3** (Required): an S3 prefix for uploading the packed and compressed model artifacts
  tar file.
- **-\-infer-task-type** (Required): the name of real-time inference task. Options include ``node_classification``
  and ``node_regression``.
- **-\-model-name**: the name of model. This name will be used to define names of SageMaker Model,
  EndpointConfig, and Endpoint by appending datetime to this model name. The name should follow a regular
  expression pattern: ``^[a-zA-Z0-9]([\-a-zA-Z0-9]*[a-zA-Z0-9])$`` as defined in
  `SageMaker's CreateEndpoint document <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEndpoint.html>`_.
  (Default: ``GSF-Model4Realtime``)

Outputs of this command include the deployed endpoint ARN (Amazon Resource Name) based on the value for
``--model-name``, e.g., ``arn:aws:sagemaker:us-east-1:123456789012:endpoint/GSF-Model4Realtime-Endpoint-2025-06-04-23-47-11``,
This endpoint name will be used in the invoke step. The endpoint ARN can also be found from Amazon SageMaker
AI Web console under the "Inference -> Endpoints" menu.

Invoke Real-time Inference Endpoints
-------------------------------------

For real-time inference, you will need to extract a subgraph around the target nodes/edges from a large
graph, and use the subgraph as input of model, which is similar to how models are trained. Because time is
critical for real-time infernce, it is recommened to use OLTP graph database, e.g.,
`Amazon Neptune Database <https://aws.amazon.com/neptune/>`_, as data source for subgraph extraction.

Once the subgraph is extracted, you will need to prepare it as the payload of different APIs for `invoke 
models for real-time inference
<https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-test-endpoints.html#realtime-endpoints-test-endpoints-api>`_.
GraphStorm defines a :ref:`specification of the payload contents <rt-request_payload-spec>` for your reference.

Below is an example payload JSON object for node classification inference.

.. code:: yaml

    {
        "version": "gs-realtime-v0.1",
        "gml_task": "node_classification",
        "graph": {
            "nodes": [
                {
                    "node_type": "author",
                    "node_id": "a4444",
                    "features": {"feat_num" : [ 0.011269339360296726, ......, ],
                                 "feat_cat" : "UK"},
                },
                {
                    "node_type": "author",
                    "node_id": "a39",
                    "features": {"feat": [-0.0032965524587780237, ......, ],
                                "feat_cat" : "USA"},
                },
                ......
            ],
            "edges": [
                {
                    "edge_type": [
                        "author",
                        "writing",
                        "paper"
                    ],
                    "src_node_id": "p4463",
                    "dest_node_id": "p4463",
                    "features": {"feat1": [ 1.411269339360296726, ......, ]},
                                 "feat2" : "1980s"},
                },
                ......
            ]
        },
        "targets": [
            {
                "node_type": "author",
                "node_id": "a39"
            }
        ]
    }

Invoke endpoints
****************
There are multiple ways to invoke a Sagemaker real-time inference endpoint as documented in
`SageMaker Developer Guide <https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-test-endpoints.html#realtime-endpoints-test-endpoints-api>`_.

.. note:: GraphStorm real-time inference endpoints currently only support "application/json" as content type.

Here is an example of how you can read a payload from a JSON file and use the boto3 APIs to
invoke an endpoint.

.. code-block:: python

    import json
    import boto3
    

    # Create a SageMaker client object\n",
    sagemaker = boto3.client('sagemaker')
    # Create a SageMaker runtime client object using your IAM role ARN\n",
    runtime = boto3.client('sagemaker-runtime',
                           aws_access_key_id='your access key string',
                           aws_secret_access_key='your secret key string',
                           region_name='asw region' # e.g., us-east-1
    endpoint_name='your endpoint name'    # e.g., GSF-Model4Realtime-Endpoint-2025-07-11-21-44-36
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

Endpoint invoke response
*************************

As shown in the previous invoke example, the response from a GraphStorm real-time inference endpoint will include
a JSON object in the ``Body`` field of the SageMaker API response. The details of the response syntax can be found
in the :ref:`specification of realt-time inference response <rt-response-body-spec>`.

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
