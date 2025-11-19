.. _real-time-inference-spec:

==========================================================
Specification of Real-time Inference Request and Response
==========================================================

.. _rt-request-payload-spec:

Specification of Request Payload Contents 
------------------------------------------

A payload should be a JSON object. In the highest level, the JSON object contains four fields:
``version``, ``gml_task``, ``graph``, and ``targets``.

.. code:: json

    {
        "version"   : string,
        "gml_task"  : string,
        "graph"     : object,
        "targets"   : [ ... ]
    }

- ``version`` -- (String, required) The version of payload to be used. The current version is ``gs-realtime-v0.1``.
- ``gml_task`` -- (String, required) The graph machine learning task this payload is for. The current version
  supports two options: 
    * ``node_classification``
    * ``node_regression``
- ``graph`` -- (JSON objects, required) The contents of a payload, with "nodes", and "edges" keys. See below for details.
- ``targets`` -- (Array of JSON objects, required) The contents of target nodes or edges for prediciton.

Contents of objects in the ``graph`` field
...........................................

A ``graph`` object contains two objects: ``nodes``, and ``edges``.

.. code:: json

    {
        "nodes"     : [ ... ],
        "edges"     : [ ... ]
    }

- ``nodes`` -- (array of JSON objects) Each object specifies a ``node`` object. 
- ``edges`` -- (array of JSON objects) Each object specifies an ``edge`` object.

Contents of a ``node`` object listed in a ``nodes`` array
**********************************************************

A ``node`` object listed in a ``nodes`` array can contain the following required fields.

.. code:: json

    {
        "node_type" : string,
        "node_id"   : string,
        "features"  : {
                        feature name: feature value,
                        feature name: feature value,
                        ...
                      }
    }

* ``node_type`` -- A string specifying the node type name in a graph. It should be same as the
  ``node_type`` defined in :ref:`GConstruct JSON specification <gconstruction-json>` or the ``type``
  values of ``nodes`` defined in :ref:`GSProcessing JSON specification <gsprocessing_input_configuration>`.
* ``node_id`` -- A string specifying a unique node identifier. The identifiers are used to build a sub-graph according to the contents of the `edges` object.
* ``features`` -- A dictionary, with feature names as keys, and feature values as its values.
  Features names must match the ``feature_name`` entries defined in :ref:`GConstruct JSON specification
  <gconstruction-json>`, or the ``name`` values of ``features`` fields defined in
  :ref:`GSProcessing JSON specification <gsprocessing_input_configuration>`.

Contents of an ``edge`` object listed in an ``edges`` array
************************************************************

An ``edge`` object listed in an ``edges`` array must contain the following required fields.

.. code:: json

    {
        "edge_type"     : [(source node type), (edge type), (destination node type)],
        "src_node_id"   : string,
        "dest_node_id"  : string,
            "features"  : {
                            feature name: feature value,
                            feature name: feature value,
                            ...
                        }
    }

* ``edge_type`` -- An array specifying the edge type name in the format of three strings, which indicate the
  source node type, the edge type, and the destination edge type. It should be same as the ``relation`` fields
  defined in :ref:`GConstruct JSON specification <gconstruction-json>` or the ``type`` values of ``source``
  ``relation``, and ``dest`` fileds defined in
  :ref:`GSProcessing JSON specification <gsprocessing_input_configuration>`.
* ``src_node_id`` -- A string specifying the source node identifier.
* ``dest_node_id`` -- A string specifying the destination node identifier.
* ``features`` -- A dictionary, with feature names as keys, and feature values as its values.
  feature names should be same as these ``feature_name`` defined in :ref:`GConstruct JSON specification
  <gconstruction-json>`, or these ``name`` values of ``features`` fields defined in
  :ref:`GSProcessing JSON specification <gsprocessing_input_configuration>`.

Contents of text features
************************************************************

If the deployed model utilizes text features for model training, in the payload users can submit raw text directly as shown below. Features names must match the ``feature_name`` entries defined in :ref:`GConstruct JSON specification
  <gconstruction-json>`, or the ``name`` values of ``features`` fields defined in
  :ref:`GSProcessing JSON specification <gsprocessing_input_configuration>`.

.. code:: json

    {
        "features"  : {
                    feature name: <feature-text-value>,
                    ...
                }
    }

Contents of a target object listed in a ``targets`` array
..........................................................

Depending on the value of ``gml_task``, a target object in a ``targets`` array could be a ``node`` object
or an ``edge`` object as defined above. As a target object, the ``features`` field is not required. 

    .. note::

        A target object, a ``node`` or an ``edge``, should have the same ``node`` or ``edge`` object
        in the ``nodes`` or ``edges`` array. For example, in the below payload example, the ``author``
        node ``a39`` is a target node, and it also is one of the nodes in the ``nodes`` list.

        .. code:: json

            {
                "version": "gs-realtime-v0.1",
                "gml_task": "node_classification",
                "graph": {
                    "nodes": [
                        {
                            "node_type": "author",
                            "node_id": "a4444",
                            "features": { ...... },
                        },
                        {
                            "node_type": "author",
                            "node_id": "a39",
                            "features": { ...... },
                        },
                        ......
                    ],
                    "edges": [ ......]
                },
                "targets": [
                    {
                        "node_type": "author",
                        "node_id": "a39"
                    }
                ]
            }

.. _rt-response-body-spec:

Specification of Response Body Contents 
----------------------------------------

A response body is a JSON object.

**Response Body Syntax**:
.........................

.. code:: json

    {
        "status_code"   : "int",
        "request_uid"   : "string",
        "message"       : "string",
        "error"         : "string",
        "data"          : {
            results: [
                {
                    "node_type"     : "string",
                    "node_id"       : "string",
                    "predictions"   : [ ...... ]
                },
                or
                {
                    "edge_type"     : ["string", "string", "string"],
                    "src_node_id"   : "string",
                    "dest_node_id"  : "string",
                    "predictions"   : [ ...... ]
                }
            ]
        }
    }

**Response Body Structure**:
............................

- (dict) --
    - ``status_code`` (int) --
        An integer indicates the outcome status, including:
            - ``200``: request processed successfully.
            - ``400``: the request payload has JSON format errors.
            - ``401``: the request payload missed certain fileds, required by :ref:`Payload specification <rt-request-payload-spec>`.
            - ``402``: the request payload missed values on certain fileds, e.g., missing a node identifier in ``node_id`` field.
            - ``403``: ``node_type`` of nodes in the ``target`` field does not exist in the ``graph`` field.
            - ``404``: values of the ``node_id`` fileds of nodes in the ``target`` field do not exist in the ``graph`` field.
            - ``411``: errors occurred when converting the request payload into DGL graph format for inference.
            - ``421``: the task in ``gml_task`` does not match the task that the deployed model is for.
            - ``500``: internal server errors.
    - ``request_uid`` (string) --
        A string serves as a unique identifier for the request payload. This identifier is logged on the
        endpoint side and returned to invokers, facilitating error debugging.
    -  ``message`` (string) --
        A string provides additional information when the ``status_code`` is 200.
    - ``error`` (string) --
        A string provides detailed explanations when the ``status_code`` is **NOT** 200.
    - ``data`` (dict) --
        When the ``status_code`` is 200, includes a populated ``data`` field. Otherwise, the ``data`` field
        is empty.
            - ``results`` (list) --
                A list that includes the inference values for all nodes or edges specified in the payload's
                ``targets`` field.
                    - (dict) --
                        For node prediction tasks (node classification and node regression):
                            - ``node_type`` (string) --
                                Specifies a node type name in a graph.
                            - ``node_id`` (string) -- 
                                Specifies a node identifier.
                        For edge prediciton tasks (edge classification and edge regression):
                            - ``edge_type`` (list ) --
                                An array specifying the edge type name in the format of three strings, which
                                indicate the source node type, the edge type, and the destination edge type.
                            - ``src_node_id`` (string) --
                                Specifies the source node identifier.
                            - ``dest_node_id`` (string) --
                                Specifies the destination node identifier.

                        - ``prediction`` (list) --
                            A list containing the inference results for all target nodes or edges. For classification
                            tasks, the value of ``prediction`` is a list of logits that can be used with classification
                            methods such as `argmax`. For regression tasks, the value of ``prediction`` is a list with
                            a single element, which represents the regression result.

