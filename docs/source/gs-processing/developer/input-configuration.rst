..  _input-configuration:

GraphStorm Processing Input Configuration
=========================================

GraphStorm Processing uses a JSON configuration file to
parse and process the data into the format needed
by GraphStorm partitioning and training downstream.

We use this configuration format as an intermediate
between other config formats, such as the one used
by the single-machine GConstruct module.

GSProcessing can take a GConstruct-formatted file
directly, and we also provide `a script <https://github.com/awslabs/graphstorm/blob/main/graphstorm-processing/scripts/convert_gconstruct_config.py>`
that can convert a `GConstruct <https://graphstorm.readthedocs.io/en/latest/configuration/configuration-gconstruction.html#configuration-json-explanations>`
input configuration file into the ``GSProcessing`` format,
although this is mostly aimed at developers, users are
can rely on the automatic conversion.

The GSProcessing input data configuration has two top-level objects:

.. code-block:: json

   {
     "version": "gsprocessing-v1.0",
     "graph": {}
   }

-  ``version`` (String, required): The version of configuration file being used. We include
   the package name to allow self-contained identification of the file format.
-  ``graph`` (JSON object, required): one configuration object that defines each
   of the node types and edge types that describe the graph.

We describe the ``graph`` object next.

``graph`` configuration object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``graph`` configuration object can have two top-level objects:

.. code-block:: json

   {
     "edges": [{}],
     "nodes": [{}]
   }

-  ``edges``: (array of JSON objects, required). Each JSON object
   in this array describes one edge type and determines how the edge
   structure will be parsed.
-  ``nodes``: (array of JSON objects, optional). Each JSON object
   in this array describes one node type. This key is optional, in case
   it is missing, node IDs are derived from the ``edges`` objects.

--------------

Contents of an ``edges`` configuration object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ``edges`` configuration object can contain the following top-level
objects:

.. code-block:: json

   {
     "data": {
       "format": "String",
       "files": ["String"],
       "separator": "String"
     },
     "source": {"column": "String", "type": "String"},
     "relation": {"type": "String"},
     "destination": {"column": "String", "type": "String"},
     "labels" : [
            {
                "column": "String",
                "type": "String",
                "split_rate": {
                    "train": "Float",
                    "val": "Float",
                    "test": "Float"
                }
            },
      ]
       "features": [{}]
   }

-  ``data`` (JSON Object, required): Describes the physical files
   that store the data described in this object. The JSON object has two
   top level objects:

   -  ``format`` (String, required): indicates the format the data is
      stored in. We accept either ``"csv"`` or ``"parquet"`` as valid
      file formats.

   -  ``files`` (array of String, required): the physical location of
      files. The format accepts two options:

      -  a single-element list a with directory-like (ending in ``/``)
         **relative** path under which all the files that correspond to
         the current edge type are stored.

         -  e.g. ``"files": ['path/to/edge/type/']``
         -  This option allows for concise listing of entire types and
            would be preferred. All the files under the path will be loaded.

      -  a multi-element list of **relative** file paths.

         -  ``"files": ['path/to/edge/type/file_1.csv', 'path/to/edge/type/file_2.csv']``
         -  This option allows for multiple types to be stored under the
            same input prefix, but will result in more verbose spec
            files.

      -  Since the spec expects **relative paths**, the caller is
         responsible for providing a path prefix to the execution
         engine. The prefix will determine if the source is a local
         filesystem or S3, allowing the spec to be portable, i.e. a user
         can move the physical files and the spec will still be valid,
         as long as the relative structure is kept.

   -  ``separator`` (String, optional): Only relevant for CSV files,
      determines the separator used between each column in the files.

-  ``source``: (JSON object, required): Describes the source nodes
   for the edge type. The top-level keys for the object are:

   -  ``column``: (String, required) The name of the column in the
      physical data files.
   -  ``type``: (String, optional) The type name of the nodes. If not
      provided, we assume that the column name is the type name.

-  ``destination``: (JSON object, required): Describes the
   destination nodes for the edge type. Its format is the same as the
   ``source`` key, with a JSON object that contains
   ``{“column: String, and ”type“: String}``.
-  ``relation``: (JSON object, required): Describes the relation
   modeled by the edges. A relation can be common among all edges, or it
   can have sub-types. The top-level objects for the object are:

   -  ``type`` (String, required): The type of the relation described by
      the edges. For example, for a source type ``user``, destination
      ``movie`` we can have a relation type ``interacted_with`` for an
      edge type ``user:interacted_with:movie``.

-  ``labels`` (List of JSON objects, optional): Describes the label
   for the current edge type. The label object has the following
   top-level objects:

   -  ``column`` (String, required): The column that contains the values
      for the label. Should be the empty string, ``""`` if the ``type``
      key has the value ``"link_prediction"``.
   -  ``type`` (String, required): The type of the learning task. Can
      take the following String values:

      -  ``“classification”``: An edge classification task. The values
         in the specified ``column`` as treated as categorical
         variables.
      -  ``"regression"``: An edge regression task. The values in the
         specified ``column`` are treated as numerical values.
      -  ``"link_prediction"``: A link prediction tasks. The ``column``
         should be ``""`` in this case.

   -  ``separator``: (String, optional): For multi-label classification
      tasks, this separator is used within the column to list multiple
      classification labels in one entry.
   -  ``split_rate`` (JSON object, optional): Defines a split rate
      for the label items. The sum of the values for ``train``, ``val`` and
      ``test`` needs to be 1.0.

      -  ``train``: The percentage of the data with available labels to
         assign to the train set (0.0, 1.0].
      -  ``val``: The percentage of the data with available labels to
         assign to the train set [0.0, 1.0).
      -  ``test``: The percentage of the data with available labels to
         assign to the train set [0.0, 1.0).

-  ``features`` (List of JSON objects, optional)\ **:** Describes
   the set of features for the current edge type. See the :ref:`features-object` section for details.

--------------

Contents of a ``nodes`` configuration object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A node configuration object in a ``nodes`` field can contain the
following top-level keys:

.. code-block:: json

    {
        "data": {
            "format": "String",
            "files": ["String"],
            "separator": "String"
        },
        "column" : "String",
        "type" : "String",
        "labels" : [
            {
                "column": "String",
                "type": "String",
                "separator": "String",
                "split_rate": {
                    "train": "Float",
                    "val": "Float",
                    "test": "Float"
                }
            }
        ],
        "features": [{}]
    }

-  ``data``: (JSON object, required): Has the same definition as for
   the edges object, with one top-level key for the ``format`` that
   takes a String value, and one for the ``files`` that takes an array
   of String values.
-  ``column``: (String, required): The column in the data that
   corresponds to the column that stores the node ids.
-  ``type:`` (String, optional): A type name for the nodes described
   in this object. If not provided the ``column`` value is used as the
   node type.
-  ``labels``: (List of JSON objects, optional): Similar to the
   labels object defined for edges, but the values that the ``type`` can
   take are different.

   -  ``column`` (String, required): The name of the column that
      contains the label values.
   -  ``type`` (String, required): Specifies that target task type which
      can be:

      -  ``"classification"``: A node classification task. The values in the specified
         ``column`` are treated as categorical variables.
      -  ``"regression"``: A node regression task. The values in the specified
         ``column`` are treated as float values.

   -  ``separator`` (String, optional): For multi-label
      classification tasks, this separator is used within the column to
      list multiple classification labels in one entry.

      -  e.g. with separator ``|`` we can have ``action|comedy`` as a
         label value.

   -  ``split_rate`` (JSON object, optional): Defines a split rate
      for the label items. The sum of the values for ``train``, ``val`` and
      ``test`` needs to be 1.0.

      -  ``train``: The percentage of the data with available labels to
         assign to the train set (0.0, 1.0].
      -  ``val``: The percentage of the data with available labels to
         assign to the train set [0.0, 1.0).
      -  ``test``: The percentage of the data with available labels to
         assign to the train set [0.0, 1.0).

-  ``features`` (List of JSON objects, optional): Describes
   the set of features for the current edge type. See the next section, :ref:`features-object`
   for details.

--------------

.. _features-object:

Contents of a ``features`` configuration object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An element of a ``features`` configuration object (for edges or nodes)
can contain the following top-level keys:

.. code-block:: json

    {
        "column": "String",
        "name": "String",
        "transformation": {
        "name": "String",
        "kwargs": {
            "arg_name": "<value>"
        }
        },
        "data": {
            "format": "String",
            "files": ["String"],
            "separator": "String"
        }
    }

-  ``column`` (String, required): The column that contains the raw
   feature values in the dataset
-  ``transformation`` (JSON object, optional): The type of
   transformation that will be applied to the feature. For details on
   the individual transformations supported see :ref:`supported-transformations`.
   If this key is missing, the feature is treated as
   a **no-op** feature without ``kwargs``.

   -  ``name`` (String, required): The name of the transformation to be
      applied.
   -  ``kwargs`` (JSON object, optional): A dictionary of parameter
      names and values. Each individual transformation will have its own
      supported parameters, described in :ref:`supported-transformations`.

-  ``name`` (String, optional): The name that will be given to the
   encoded feature. If not given, **column** is used as the output name.
-  ``data`` (JSON object, optional): If the data for the feature
   exist in a file source that's different from the rest of the data of
   the node/edge type, they are provided here. For example, you could
   have each feature in one file source each:

   .. code-block:: python

        # Example node config with multiple features
        {
            # This is where the node structure data exist just need an id col
            "data": {
                "format": "parquet",
                "files": ["path/to/node_ids"]
            },
            "column" : "node_id",
            "type" : "my_node_type",
            "features": [
                # Feature 1
                {
                    "column": "feature_one",
                    # The files contain one "node_id" col and one "feature_one" col
                    "data": {
                        "format": "parquet",
                        "files": ["path/to/feature_one/"]
                    }
                },
                # Feature 2
                {
                    "column": "feature_two",
                    # The files contain one "node_id" col and one "feature_two" col
                    "data": {
                        "format": "parquet",
                        "files": ["path/to/feature_two/"]
                    }
                }
            ]
        }


   **The file source needs
   to contain the column names of the parent node/edge type to allow a
   1-1 mapping between the structure and feature files.**

   For nodes the
   the feature files need to have one column named with the node id column
   name, (the value of ``"column"`` for the parent node type),
   for edges we need both the ``source`` and
   ``destination`` columns to use as a composite key.

.. _supported-transformations:

Supported transformations
~~~~~~~~~~~~~~~~~~~~~~~~~

In this section we'll describe the transformations we support.
The name of the transformation is the value that would appear
in the ``transform['name']`` element of the feature configuration,
with the attached ``kwargs`` for the transformations that support
arguments.

-  ``no-op``

   -  Passes along the data as-is to be written to storage and
      used in the partitioning pipeline. The data are assumed to be single
      values or vectors of floats.
   -  ``kwargs``:

      -  ``separator`` (String, optional): Only relevant for CSV file
         sources, when a separator is used to encode vector feature
         values into one column. If given, the separator will be used to
         split the values in the column and create a vector column
         output. Example: for a separator ``'|'`` the CSV value
         ``1|2|3`` would be transformed to a vector, ``[1, 2, 3]``.

--------------

Examples
~~~~~~~~

OAG-Paper dataset
-----------------

.. code-block:: json

    {
        "version" : "gsprocessing-v1.0",
        "graph" : {
            "edges" : [
                {
                "data": {
                    "format": "csv",
                    "files": [
                        "edges.csv"
                    ],
                    "separator": ","
                },
                "source": {"column": "~from", "type": "paper"},
                "dest": {"column": "~to", "type": "paper"},
                "relation": {"type": "cites"}
                }
            ],
            "nodes" : [
                {
                    "data": {
                        "format": "csv",
                        "separator": ",",
                        "files": [
                            "node_feat.csv"
                        ]
                    },
                    "type": "paper",
                    "column": "ID",
                    "labels": [
                        {
                            "column": "field",
                            "type": "classification",
                            "separator": ";",
                            "split_rate": {
                                "train": 0.7,
                                "val": 0.1,
                                "test": 0.2
                            }
                        }
                    ]
                }
            ]
        }
    }
