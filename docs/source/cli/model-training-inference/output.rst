.. _gs-output:

GraphStorm Training and Inference Output
========================================
GraphStorm training pipeline can save both trained model checkpoints and node embeddings
on disk. When ``save_model_path`` is provided in the training configuration,
the trained model checkpoints will be saved in the corresponding path.
The contents of the ``save_model_path`` will look like following:

.. code-block:: bash

    model_dir/
        epoch-0-iter-1099/
        epoch-0-iter-2099/
        epoch-0/
        epoch-1-iter-1099/
        epoch-1-iter-2099/
        ...

When ``save_embed_path`` is provided in the training configuration,
the node embeddings produced by the best model checkpoint will be saved
in the corresponding path. When the training task is launched by
GraphStorm CLIs, a node ID remapping process will be launched
automatically, after the training job, to process the saved node embeddings and the corresponding node IDs. The final output of node
embeddings will be in parquet format by default. Details can be found in :ref:`GraphStorm Output Node ID Remapping<gs-output-remapping>`

GraphStorm inference pipeline can save both node embeddings and prediction
results on disk. When ``save_embed_path`` is provided in the inference configurations,
the node embeddings will be saved in the same way as GraphStorm training pipeline.
When ``save_prediction_path`` is provided in the inference configurations,
GraphStorm will save the prediction results in the corresponding path.
When the inference task is launched by GraphStorm CLIs, a ndoe ID remapping
process will be launched automatically, after the inference job, to
process the saved prediction results and the corresponding node IDs.
The final output of prediction results will be in parquet format by default.
Details can be found in :ref:`GraphStorm Output Node ID Remapping<gs-output-remapping>`


The following sections will introduce how the node embeddings and prediction
results are saved by the GraphStorm training and inference scripts.

.. note::

    In most of the end-to-end training and inference cases, the saved files, usually in ``.pt`` format, are not consumable by the downstream applications. The :ref:`GraphStorm Output Node ID Remapping<gs-output-remapping>` must be invoked to process the output files.


.. _gs-output-embs:

Saved Node Embeddings
---------------------
When ``save_embed_path`` is provided in the training configuration or the inference configuration,
GraphStorm will save the node embeddings in the corresponding path. The node embeddings
of each node type are saved separately under different sub-directories named with
the corresponding node types. GraphStorm will also save an ``emb_info.json`` file,
which contains all the metadata for the saved node embeddings.
The contents of the ``save_embed_path`` will look like following:

.. code-block:: bash

    emb_dir/
        ntype0/
            embed_nids-00000.pt
            embed_nids-00001.pt
            ...
            embed-00000.pt
            embed-00001.pt
            ...
        ntype1/
            embed_nids-00000.pt
            embed_nids-00001.pt
            ...
            embed-00000.pt
            embed-00001.pt
            ...
        ...
        emb_info.json

The ``embed_nids-*`` files store the integer node IDs of each node embedding and
the ``embed-*`` files store the corresponding node embeddings.
The contents of ``embed_nids-*`` files and ``embed-*`` files look like:

.. code-block::

    embed_nids-00000.pt  |   embed-00000.pt
                         |
    Graph Node ID        |   embeddings
    10                   |   0.112,0.123,-0.011,...
    1                    |   0.872,0.321,-0.901,...
    23                   |   0.472,0.432,-0.732,...
    ...

The ``emb_info.json`` stores three types of information:
  * ``format``: The format of the saved embeddings. By default, it is ``pytorch``.
  * ``emb_name``: A list of node types that have node embeddings saved. For example: ["ntype0", "ntype1"]
  * ``world_size``: The number of chunks (files) into which the node embeddings of a particular node type are divided. For instance, if world_size is set to 8, there will be 8 files for each node type's node embeddings."

**Note: The built-in GraphStorm training or inference pipeline
(launched by GraphStorm CLIs) will process the saved node embeddings
to convert the integer node IDs into the raw node IDs, which are usually
string node IDs. The final output will be in parquet format by default.
And the node embedding files, i.e.,``embed-*.pt`` files, and node ID
files, i.e.,``embed_nids-*.pt`` files, will be removed.** Details can be
found in :ref:`GraphStorm Output Node ID Remapping<gs-output-remapping>`

.. _gs-out-predictions:

Saved Prediction Results
------------------------
When ``save_prediction_path`` is provided in the inference configurations,
GraphStorm will save the prediction results in the corresponding path.
For node prediction tasks, the prediction results are saved per node type.
GraphStorm will also save an ``result_info.json`` file, which contains all
the metadata for the saved prediction results. The contents of the ``save_prediction_path``
will look like following:

.. code-block:: bash

    prediction_dir/
        ntype0/
            predict-00000.pt
            predict-00001.pt
            ...
            predict_nids-00000.pt
            predict_nids-00001.pt
            ...
        ntype1/
            predict-00000.pt
            predict-00001.pt
            ...
            predict_nids-00000.pt
            predict_nids-00001.pt
            ...
        ...
        result_info.json

The ``predict_nids-*`` files store the integer node IDs of each prediction result and
the ``predict-*`` files store the corresponding prediction results.
The content of ``predict_nids-*`` files and ``predict-*`` files looks like:

.. code-block::

    predict_nids-00000.pt  |   predict-00000.pt
                           |
    Graph Node ID          |   Prediction results
    10                     |   0.112
    1                      |   0.872
    23                     |   0.472
    ...

The ``result_info.json`` stores three types of information:
  * ``format``: The format of the saved prediction results. By default, it is ``pytorch``.
  * ``emb_name``: A list of node types that have node prediction results saved. For example: ["ntype0", "ntype1"]
  * ``world_size``: The number of chunks (files) into which the prediction results of a particular node type are divided. For instance, if world_size is set to 8, there will be 8 files for each node type's prediction results."


For edge prediction tasks, the prediction results are saved per edge type.
The sub-directory for an edge type is named as ``<src_ntype>_<relation_type>_<dst_ntype>``.
For instance, given an edge type ``("movie","rated-by","user")``, the corresponding
sub-directory is named as ``movie_rated-by_user``.
GraphStorm will also save an ``result_info.json`` file, which contains all
the metadata for the saved prediction results. The contents of the ``save_prediction_path``
will look like following:

.. code-block:: bash

    prediction_dir/
        etype0/
            predict-00000.pt
            predict-00001.pt
            ...
            src_nids-00000.pt
            src_nids-00001.pt
            ...
            dst_nids-00000.pt
            dst_nids-00001.pt
            ...
        etype1/
            predict-00000.pt
            predict-00001.pt
            ...
            src_nids-00000.pt
            src_nids-00001.pt
            ...
            dst_nids-00000.pt
            dst_nids-00001.pt
            ...
        ...
        result_info.json

The ``src_nids-*`` and ``dst_nids-*`` files contain the integer node IDs for
the source and destination nodes of each prediction, respectively.
The ``predict-*`` files store the corresponding prediction results.
The content of ``src_nids-*``, ``dst_nids-*`` and ``predict-*`` files looks like:

.. code-block::

    src_nids-00000.pt   |   dst_nids-00000.pt   |   predict-00000.pt
                        |
    Source Node ID      |   Destination Node ID |   Prediction results
    10                  |   12                  |   0.112
    1                   |   20                  |   0.872
    23                  |   3                   |   0.472
    ...

The ``result_info.json`` stores three types of informations:
  * ``format``: The format of the saved prediction results. By default, it is ``pytorch``.
  * ``etypes``: A list of edge types that have edge prediction results saved. For example: [("movie","rated-by","user"), ("user","watched","movie")]
  * ``world_size``: The number of chunks (files) into which the prediction results of a particular edge type are divided. For instance, if world_size is set to 8, there will be 8 files for each edge type's prediction results."

**Note: The built-in GraphStorm inference pipeline
(launched by GraphStorm CLIs) will process the saved prediction results
to convert the integer node IDs into the raw node IDs, which are usually string node IDs. The final output will be in parquet format by default.
And the prediction files, i.e.,``predict-*.pt`` files, and node ID files,
i.e.,``predict_nids-*.pt``, ``src_nids-*.pt``, and ``dst_nids-*.pt`` files
will be removed.** Details can be found in :ref:`GraphStorm Output Node ID Remapping<gs-output-remapping>`
