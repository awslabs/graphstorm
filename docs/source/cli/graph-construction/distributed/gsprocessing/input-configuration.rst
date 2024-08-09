..  _gsprocessing_input_configuration:

GSProcessing Input Configuration
================================

GraphStorm Processing uses a JSON configuration file to
parse and process the data into the format needed
by GraphStorm partitioning and training downstream.

We use this configuration format as an intermediate
between other config formats, such as the one used
by the single-machine GConstruct module.

GSProcessing can take a GConstruct-formatted file
directly, and we also provide `a script <https://github.com/awslabs/graphstorm/blob/main/graphstorm-processing/scripts/convert_gconstruct_config.py>`_
that can convert a `GConstruct <https://graphstorm.readthedocs.io/en/latest/configuration/configuration-gconstruction.html#configuration-json-explanations>`_
input configuration file into the ``GSProcessing`` format,
although this is mostly aimed at developers, users are
can rely on the automatic conversion.

The GSProcessing input data configuration has two top-level objects:

.. code-block:: json

   {
     "version": "gsprocessing-v0.3.1",
     "graph": {}
   }

-  ``version`` (String, required): The version of configuration file being used. We include
   the package name to allow self-contained identification of the file format.
-  ``graph`` (JSON object, required): one configuration object that defines each
   of the edge and node types that constitute the graph.

We describe the ``graph`` object next.

Contents of the ``graph`` configuration object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
     "dest": {"column": "String", "type": "String"},
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
      ],
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
   modeled by the edges. The top-level keys for the object are:

   -  ``type`` (String, required): The type of the relation described by
      the edges. For example, for a source type ``user``, destination
      ``movie`` we can have a relation type ``rated`` for an
      edge type ``user:rated:movie``.

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
         assign to the validation set [0.0, 1.0).
      -  ``test``: The percentage of the data with available labels to
         assign to the test set [0.0, 1.0).
   -  ``custom_split_filenames`` (JSON object, optional): Specifies the customized
      training/validation/test mask. Once it is defined, GSProcessing will ignore
      the ``split_rate``.
      -  ``train``: Path of the training mask parquet file such that each line contains
    the original ID for node tasks, or the pair [source_id, destination_id] for edge tasks.
      -  ``val``: Path of the validation mask parquet file such that each line contains
    the original ID for node tasks, or the pair [source_id, destination_id] for edge tasks.
      -  ``test``: Path of the test mask parquet file such that each line contains
    the original ID for node tasks, or the pair [source_id, destination_id] for edge tasks.

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
        "column": "String",
        "type": "String",
        "labels" : [
            {
                "column": "String",
                "type": "String",
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
-  ``column``: (String, required): The name of the column in the data that
   stores the node ids.
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
         assign to the validation set [0.0, 1.0).
      -  ``test``: The percentage of the data with available labels to
         assign to the test set [0.0, 1.0).

-  ``features`` (List of JSON objects, optional): Describes
   the set of features for the current node type. See the section :ref:`features-object`
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
   feature values in the data.
-  ``transformation`` (JSON object, optional): The type of
   transformation that will be applied to the feature. For details on
   the individual transformations supported see :ref:`gsp-supported-transformations-ref`.
   If this key is missing, the feature is treated as
   a **no-op** feature without ``kwargs``.

   -  ``name`` (String, required): The name of the transformation to be
      applied.
   -  ``kwargs`` (JSON object, optional): A dictionary of parameter
      names and values. Each individual transformation will have its own
      supported parameters, described in :ref:`gsp-supported-transformations-ref`.

-  ``name`` (String, optional): The name that will be given to the
   encoded feature. If not given, **column** is used as the output name.
-  ``data`` (JSON object, optional): If the data for the feature
   exist in a file source that's different from the rest of the data of
   the node/edge type, they are provided here. For example, you could
   have each feature in one file source each:

   .. code-block:: python

        # Example node config with multiple features
        {
            # This is where the node structure data exist, just need an id col in these files
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

.. _gsp-supported-transformations-ref:

Supported transformations
~~~~~~~~~~~~~~~~~~~~~~~~~

In this section we'll describe the transformations we support.
The name of the transformation is the value that would appear
in the ``['transformation']['name']`` element of the feature configuration,
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
      - ``truncate_dim`` (Integer, Optional): Relevant for vector inputs.
        Allows you to truncate the input vector to the first ``truncate_dim``
        values, which can be useful when your inputs are `Matryoshka representation
        learning embeddings <https://arxiv.org/abs/2205.13147>`_.
      - ``out_dtype`` (String, Optional): Specify the data type of the transformed feature.
        Currently we only support ``float32`` and ``float64`` .
-  ``numerical``

   -  Transforms a numerical column using a missing data imputer and an
      optional normalizer.
   -  ``kwargs``:

      -  ``imputer`` (String, optional): A method to fill in missing values in the data.
         Valid values are:
         ``none`` (Default), ``mean``, ``median``, and ``most_frequent``. Missing values will be replaced
         with the respective value computed from the data.
      - ``normalizer`` (String, optional): Applies a normalization to the data, after imputation.
        Can take the following values:

         - ``none``: (Default) Don't normalize the numerical values during encoding.
         - ``min-max``: Normalize each value by subtracting the minimum value from it,
           and then dividing it by the difference between the maximum value and the minimum.
         - ``standard``: Normalize each value by dividing it by the sum of all the values.
         - ``rank-gauss``: Normalize each value using Rank-Gauss normalization. Rank-gauss first ranks all values,
           converts the ranks to the -1/1 range, and applies the `inverse of the error function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfinv.html>`_ to make the values conform
           to a Gaussian distribution shape. This transformation only supports a single column as input.
      - ``out_dtype`` (String, Optional): Specify the data type of the transformed feature.
        Currently we only support ``float32`` and ``float64`` .
      - ``epsilon``: Only relevant for ``rank-gauss``, this epsilon value is added to the denominator
        to avoid infinite values during normalization.
-  ``multi-numerical``

   -  Column-wise transformation for vector-like numerical data using a missing data imputer and an
      optional normalizer.
   -  ``kwargs``:

      - ``imputer`` (String, optional): Same as for ``numerical`` transformation, will
        apply no imputation by default.
      - ``normalizer`` (String, optional): Same as for ``numerical`` transformation, no
        normalization is applied by default.
      - ``separator`` (String, optional): Same as for ``no-op`` transformation, used to separate numerical
        values in CSV input. If the input data are in Parquet format, each value in the
        column is assumed to be an array of floats.
      - ``out_dtype`` (Optional): Specify the data type of the transformed feature.
        Currently we only support ``float32`` and ``float64`` .

-  ``bucket-numerical``

   -  Transforms a numerical column to a one-hot or multi-hot bucket representation, using bucketization.
      Also supports optional missing value imputation through the `imputer` kwarg.
   -  ``kwargs``:

      - ``imputer`` (String, optional): A method to fill in missing values in the data.
        Valid values are:
        ``none`` (Default), ``mean``, ``median``, and ``most_frequent``. Missing values will be replaced
        with the respective value computed from the data.
      - ``range`` (List[float], required), The range defines the start and end point of the buckets with ``[a, b]``. It should be
        a list of two floats. For example, ``[10, 30]`` defines a bucketing range between 10 and 30.
      - ``bucket_cnt`` (Integer, required), The count of bucket lists used in the bucket feature transform. GSProcessing
        calculates the size of each bucket as  ``( b - a ) / c`` , and encodes each numeric value as the number
        of whatever bucket it falls into. Any value less than a is considered to belong in the first bucket,
        and any value greater than b is considered to belong in the last bucket.
      - ``slide_window_size`` (Integer, optional), slide_window_size can be used to make numeric values fall into more than one bucket,
        by specifying a slide-window size ``s``, where ``s`` can an integer or float. GSProcessing then transforms each
        numeric value ``v`` of the property into a range from ``v - s/2`` through ``v + s/2`` , and assigns the value v
        to every bucket that the range covers.

-  ``categorical``

   -  Transforms values from a fixed list of possible values (categorical features) to a one-hot encoding.
      The length of the resulting vector will be the number of categories in the data minus one, with a 1 in
      the index of the single category, and zero everywhere else.

.. note::
    The maximum number of categories in any categorical feature is 100. If a property has more than 100 categories of value,
    only the most common 99 of them are placed in distinct categories, and the rest are placed in a special category named OTHER.

-  ``multi-categorical``

   -  Encodes vector-like data from a fixed list of possible values (i.e. multi-label/multi-categorical data) using a multi-hot encoding. The length of the resulting vector will be the number of categories in the data minus one, and each value will have a 1 value for every category that appears, and 0 everwhere else.
   -  ``kwargs``:

      - ``separator`` (String, optional): Same as the one in the No-op operation, the separator is used to
        split multiple input values for CSV files e.g. ``detective|noir``. If it is not provided, then the whole value
        will be considered as an array. For Parquet files, if the input type is ArrayType(StringType()), then the
        separator is ignored; if it is StringType(), it will apply same logic as in CSV.

-  ``huggingface``

   -  Transforms a text feature column to tokens or embeddings with different Hugging Face models, enabling nuanced understanding and processing of natural language data.
   -  ``kwargs``:

      - ``action`` (String, required): Currently we support embedding creation using HuggingFace models, where the input text is transformed to a vector representation,
        or tokenization of text the using using HuggingFace tokenizers, where the output is a tokenized version of the text to be used downstream as input to a Huggingface model during training.

        - ``tokenize_hf``: Tokenize text strings with a HuggingFace tokenizer. The tokenizer_hf can use any HuggingFace LM models available in the
          `huggingface model repository <https://huggingface.co/models>`_.
          You can find more information about tokenization at `huggingface autotokenizer docs <https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer>`_
          The expected input are text strings, and the expected output will include ``input_ids`` for token IDs on the input text,
          ``attention_mask`` for a mask to avoid performing attention on padding token indices, and ``token_type_ids`` for segmenting two sentences in models.
          The output here is compatible for graphstorm language model training and inference pipelines.

        - ``embedding_hf``: Encode text strings with a HuggingFace embedding model. The value can be any HuggingFace language model available in the
          `Huggingface model repository <https://huggingface.co/models>`_, e.g. ``bert-base-uncased``.
          The expected input are text strings, and the expected output will be the vector embeddings for the text strings.
      - ``hf_model`` (String, required): An identifier of a pre-trained model available in the Hugging Face Model Hub, e.g. ``bert-base-uncased``.
        You can find all models in the `Huggingface model repository <https://huggingface.co/models>`_.
      - ``max_seq_length`` (Integer, required): Specifies the maximum number of tokens of the input.
        You can use a length greater than the dataset's longest sentence; or for a safe value choose 128. Make sure to check
        the model's max suported length when setting this value,

--------------

Creating a graph for inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If no label entries are provided for any of the entries
in the input configuration, the processed data will not
include any train/val/test masks. You can use this mode
when you want to produce a graph just for inference.

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
                    "type": "paper",
                    "column": "ID",
                    "data": {
                        "format": "csv",
                        "separator": ",",
                        "files": [
                            "node_feat.csv"
                        ]
                    },
                    "features": [
                        {
                            "column": "n_citation",
                            "transformation": {
                                "name": "numerical",
                                "kwargs": {
                                    "imputer": "mean",
                                    "normalizer": "min-max"
                                }
                            }
                        }
                    ],
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
