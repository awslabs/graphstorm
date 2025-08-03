.. _real-time-payload-spec:

Specification of Real-time Inference Payload Contents
------------------------------------------------------

The payload should be a JSON object. In the highest level, the JSON object contains three fields:
``version``, ``gml_task``, and ``graph``.

.. code:: json

    {
        "version"   : string,
        "gml_task"  : string,
        "graph"     : [ ... ]
    }

- ``version`` -- (String) The version of payload to be used. The current version is ``gs-realtime-v0.1``.
- ``gml_task`` -- (String) The graph machine learning task this payload is for. Current specification
  supports two options: 
    * ``node_classification``
    * ``node_regression``
- ``graph`` -- (JSON objects) The contents of a payload.


Contents of objects in the ``graph`` field
........................................... 

A ``graph`` object contains three objects, i.e., ``nodes``, ``edges``, and ``targets``.

.. code:: json

    {
        "nodes"     : [ ... ],
        "edges"     : [ ... ],
        "targets"   : [ ... ]
    }

- ``nodes`` -- (array of JSON objects) Each object specifies a ``node`` object. 
- ``edges`` -- (array of JSON objects) Each object specifies an ``edge`` object.
- ``targets``  -- (array of JSON objects) Each object specifies a ``node`` object or an ``edge`` object,
  depending on the value of ``gml_task``.

Contents of a ``node`` object listed in a ``nodes`` array
..........................................................

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

* ``node_type`` -- A string specifying the node type name in a graph. It should be same as these
  ``node_type`` defined in :ref:`gconstruct JSON specification <gconstruction-json>` or the ``type``
  values of ``nodes`` defined in :ref:`gsprocessing JSON specification <gsprocessing_input_configuration>`.
* ``node_id`` -- A string specifying the node identifier.
* ``features`` -- A dictionary, whose key is a feature name, and its value is the value of features.
  feaure names should be same as the ``feature_name`` defined in :ref:`gconstruct JSON specification
  <gconstruction-json>`, or these ``name`` values of ``features`` fields defined in
  :ref:`gsprocessing JSON specification <gsprocessing_input_configuration>`.

Contents of an ``edge`` object listed in an ``edges`` array
............................................................

An ``edge`` object listed in an ``edges`` array can contain the following required fields.

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

* ``edge_type`` -- An array specifying the edge type name in the format of three strings, which indicate
  source node type, edge type, and destination edge type. It should be same as the ``relation`` fileds defined
  in :ref:`gconstruct JSON specification <gconstruction-json>` or the ``type`` values of ``source``
  ``relation``, and ``dest`` fileds defined in
  :ref:`gsprocessing JSON specification <gsprocessing_input_configuration>`.
* ``src_node_id`` -- A string specifying the source node identifier.
* ``dest_node_id`` -- A string specifying the destination node identifier.
* ``features`` -- A dictionary, whose key is a feature name, and its key is value of the feature. 
  feaure names should be same as these ``feature_name`` defined in :ref:`gconstruct JSON specification
  <gconstruction-json>`, or these ``name`` values of ``features`` fields defined in
  :ref:`gsprocessing JSON specification <gsprocessing_input_configuration>`.

Contents of a target object listed in a ``targets`` array
..........................................................

Depending on the value of ``gml_task``, a target object in a ``targets`` array could be a ``node`` object
or an ``edge`` object defined above. As a target object, the ``features`` field is not required. 

    .. note::

        A target objects, a ``node`` or an ``edge``, should have a same ``node`` or ``edge`` object
        in the ``nodes`` or ``edges`` array. For example,

        .. code:: yaml

            {
                "version": "gs-realtime-v0.1",
                "gml_task": "node_classification",
                "graph": {
                    "nodes": [
                        {
                            "node_type": "author",
                            "node_id": "a4444",
                            "features": { "feat": [ 0.011269339360296726, ......, ]},
                        },
                        {
                            "node_type": "author",
                            "node_id": "a39",
                            "features": { "feat": [-0.0032965524587780237, ......, ]},
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