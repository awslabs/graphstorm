.. _real-time-payload-spec:

The specification of Real-time Inference Payload Contents
----------------------------------------------------------

The payload should be a JSON object. In the highest level, the JSON object contains three fields:
``version``, ``gml_task``, and ``graph``.

.. code: json

    {
        "version"   : string,
        "gml_task"  : string,
        "graph"     : [ ... ]
    }

- ``version`` -- (String) The version of payload to be used. The current version is ``gs-realtime-v0.1``.
- ``gml_task`` -- (String) The graph machine learning task this payload is for. Current specification supports two
options: 

    * ``node_classification``
    * ``node_regression``

- ``graph`` (JSON objects) The contents of a payload.


Contents of objects in the ``graph`` field
........................................... 

A ``graph`` object contains three objects, i.e., ``nodes``, ``edges``, and ``targets``.


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
