.. _gs-output-remapping:

GraphStorm Output Node ID Remapping
====================================
During :ref:`Graph Construction<graph-construction>`, GraphStorm converts
user provided node IDs into integer-based node IDs. Thus, the outputs of
GraphStorm training and inference jobs, i.e., :ref:`saved node
embeddings<gs-output-embs>` and :ref:`saved prediction results<gs-out-predictions>`,
are stored with their integer-based node IDs. GraphStorm provides a
``gconstruct.remap_result`` module to remap the integer-based node IDs
back to the original user provided node IDs according to the :ref:`node ID
mapping files<gs-id-mapping-files>`.

.. note::

    If the training or inference tasks are launched by GraphStorm CLIs,
    the ``gconstruct.remap_result`` module is automatically triggered to
    to remap the integer-based node IDs back to the original user provided
    node IDs.

Output Node Embeddings after Remapping
--------------------------------------
By default, the output node embeddings after ``gconstruct.remap_result``
are stored in the path specified by ``save_embed_path`` in parquet format.
The node embeddings for different node
types are stored in separate directories, each named after the
corresponding node type. The content of the output directory will look like following:

.. code-block:: bash

    emb_dir/
        ntype0:
            embed-00000_00000.parquet
            embed-00000_00001.parquet
            ...
        ntype1:
            embed-00000_00000.parquet
            embed-00000_00001.parquet
            ...

For multi-task learning tasks, the output node embeddings may have
task specific versions. (Details can be found in :ref:`Multi-task
Learning Output<multi-task-learning-output>`). The task specific
node embeddings are also processed by the ``gconstruct.remap_result`` module.
The content of the output directory will look like following:

.. code-block:: bash

    emb_dir/
        ntype0/
            embed-00000_00000.parquet
            embed-00000_00001.parquet
            ...
        ntype1:
            embed-00000_00000.parquet
            embed-00000_00001.parquet
            ...
        link_prediction-paper_cite_paper/
            ntype0/
                embed-00000_00000.parquet
                embed-00000_00001.parquet
                ...
            ntype1:
                embed-00000_00000.parquet
                embed-00000_00001.parquet
                ...
        edge_regression-paper_cite_paper-year/
            ntype0/
                embed-00000_00000.parquet
                embed-00000_00001.parquet
                ...
            ntype1:
                embed-00000_00000.parquet
                embed-00000_00001.parquet
                ...

The content of each parquet file will look like following:

+-----+------------------------------------------------------------------+
| nid |                             emb                                  |
+=====+==================================================================+
| n0  | [0.2964, 0.0779, 1.2763, 2.8971, ..., -0.2564, 0.9060, -0.8740]  |
+-----+------------------------------------------------------------------+
| n1  | [1.6941, -1.6765, 0.1862, -0.4449, ..., 0.6474, 0.2358, -0.5952] |
+-----+------------------------------------------------------------------+
| n10 | [-0.8417, 2.5096, -0.0393, -0.8208, ..., 0.9894, 2.3389, 0.9778] |
+-----+------------------------------------------------------------------+

.. note::

    ``gconstruct.remap_result`` uses ``nid`` as the default column name
    for node IDs and ``emb`` as the default column name for embeddings


Output Prediction Results after Remapping
-----------------------------------------
By default, the output prediction results after ``gconstruct.remap_result``
are stored in the path specified by ``save_prediction_path`` in parquet format.
The prediction results for different node
types are stored in separate directories, each named after the
corresponding node type. The prediction results for different edge
types are stored in separate directories, each named after the
corresponding edge type. The content of the directory of node prediction results will look like following:

.. code-block:: bash

    predict_dir/
        ntype0:
            predict-00000_00000.parquet
            predict-00000_00001.parquet
            ...
        ntype1:
            predict-00000_00000.parquet
            predict-00000_00001.parquet
            ...

The content of the directory of edge prediction results will look like following:

.. code-block:: bash

    predict_dir/
        etype0:
            predict-00000_00000.parquet
            predict-00000_00001.parquet
            ...
        etype1:
            predict-00000_00000.parquet
            predict-00000_00001.parquet
            ...

For multi-task learning tasks, there can be multiple prediction results
for different tasks. (Details can be found in :ref:`Multi-task
Learning Output<multi-task-learning-output>`). The task specific
prediction results are also processed by the ``gconstruct.remap_result`` module.
The content of the output directory will look like following:

.. code-block:: bash

    prediction_dir/
        edge_regression-paper_cite_paper-year/
            paper_cite_paper/
                predict-00000_00000.parquet
                predict-00000_00001.parquet
                ...
        node_classification-paper-venue/
            paper/
                predict-00000_00000.parquet
                predict-00000_00001.parquet
        ...

The content of a node prediction result file will look like following:

+-----+------------------+
| nid |     pred         |
+=====+==================+
| n0  | [0.2964, 0.7036] |
+-----+------------------+
| n1  | [0.1862, 0.8138] |
+-----+------------------+
| n10 | [0.9778, 0.0222] |
+-----+------------------+

.. note::

    ``gconstruct.remap_result`` uses ``nid``as the default column name
    for node IDs and ``pred``as the default column name for prediction results.

The content of an edge prediction result file will look like following:

+---------+---------+------------------+
| src_nid | dst_nid |       pred       |
+=========+=========+==================+
|    n0   |   n32   | [0.2964, 0.7036] |
+---------+---------+------------------+
|    n1   |   n21   | [0.1862, 0.8138] |
+---------+---------+------------------+
|    n10  |   n2    | [0.9778, 0.0222] |
+---------+---------+------------------+

.. note::

    ``gconstruct.remap_result`` uses ``src_nid``as the default column name
    for source node IDs, ``dst_nid``as the default column name for
    destination node IDs and ``pred``as the default column name for prediction results.

Run remap_result Command
-------------------------
If users want to run remap_result by themselves, they can run the
``gconstruct.remap_result`` command by following the command example:

.. code:: python

    python -m graphstorm.gconstruct.remap_result \
        --node-id-mapping PATH_TO/id_mapping \
        --pred-ntypes "n0" "n1" \
        --prediction-dir PATH_TO/pred/ \
        --node-emb-dir PATH_TO/emb/ \

This example provides the actual Python command. It will do node ID
remapping for prediction results of node type `n0` and `n1`` stored
under `PATH_TO/pred/`. It will also do node ID remapping for node
embeddings stored under `PATH_TO/emb/`. The remapped data will be saved
in the save directory as the input data and the input data will be
removed to save disk space.

Below lists the full argument list of the ``gconstruct.remap_result`` command:

* **-\-node-id-mapping**: (**Required**) the path storing the node ID mapping files.
* **-\-cf**: the path to the yaml configuration file of the corresponding training or inference task. By providing the configuration file, ``gconstruct.remap_result`` will automatically infer the necessary information for ID remappings for node embeddings and prediction results.
* **-\-num-processes**: The number of processes to process the data simultaneously. A larger number of processes will speedup the ID remapping progress but consumes more CPU memory. Default is 4.
* **-\-node-emb-dir**: The directory storing the node embeddings to be remapped. Default is None.
* **-\-prediction-dir**: The directory storing the graph prediction results to be remapped. Default is None.
* **-\-pred-etypes**: A list of canonical edge types which have prediction results to be remmaped. For example, ``--pred-etypes user,rate,movie user,watch,movie``. Must be used with ``--prediction-dir``. Default is None.
* **-\-pred-ntypes**: A list of node types which have prediction results to be remmaped. For example, ``--pred-ntypes user movie``. Must be used with ``--prediction-dir``. Default is None.
* **-\-output-format**: The output format. It can be ``parquet`` or ``csv``. Default is ``parquet``.
* **-\-output-delimiter**: The delimiter used when **-\-output-format** set to ``csv``. Default is ``,``.
* **-\-column-names**: Defines how to rename default column names to new names. For example, given ``--column-names nid,~id emb,embedding``, the column ``nid``will be renamed to ``~id`` and the column ``emb`` will be renamed to `embedding`. Default is None.
* **-\-logging-level**: The logging level. The possible values: `debug`, ``info``, ``warning``, ``error``. Default is ``info``.
* **-\-output-chunk-size**: Number of rows per output file. ``gconstruct.remap_result`` will automatically split output file into multiple files. By default, it is set to ``sys.maxsize``
* **-\-preserve-input**: Whether we preserve the input data. This is only for debug purpose. Default is False.
