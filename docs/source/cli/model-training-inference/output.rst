.. _gs-output:

GraphStorm Output
=================

.. _gs-output-embs:

Saved Node Embeddings
---------------------
When ``save_embed_path`` is provided in the training config or inference condig,
GraphStorm will save the node embeddings in the corresponding path. The node embeddings
of each node type are saved separately under different sub-directories named with
the corresponding node types. GraphStorm will also save an ``emb_info.json`` file,
which contains all the metadata for the saved node embeddings. The ``save_embed_path``
will look like following:

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
The content of ``embed_nids-*`` files and ``embed-*`` files looks like:

.. code-block::

    embed_nids-00000.pt  |   embed-00000.pt
                         |
    Graph Node ID        |   embeddings
    10                   |   0.112,0.123,-0.011,...
    1                    |   0.872,0.321,-0.901,...
    23                   |   0.472,0.432,-0.732,...
    ...

The ``emb_info.json`` stores three informations:
  * ``format``: The format of the saved embeddings. By default, it is ``pytorch``.
  * ``emb_name``: A list of node types that have node embeddings saved. For example: ["ntype0", "ntype1"]
  * ``world_size``: The number of chunks (files) into which the node embeddings of a particular node type are divided. For instance, if world_size is set to 8, there will be 8 files for each set of node embeddings."

**Note: The built-in GraphStorm training or inference pipeline
(launched by GraphStorm CLI) will process the saved node embeddings
to convert the integer node ids into the raw node ids, which are usually string node ids..**
Details can be found in :ref:`GraphStorm Output Node ID Remapping<output-remapping>`

.. _gs-output-predictions:

Saved Prediction Results
------------------------
When ``save_prediction_path`` is provided in the inference condig,
GraphStorm will save the prediction results in the corresponding path.
For node prediction tasks, the prediction results are saved per node type.
GraphStorm will also save an ``result_info.json`` file, which contains all
the metadata for the saved prediction results. The ``save_prediction_path``
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

    predict_nids-00000.pt  |   predict.pt
                           |
    Graph Node ID          |   Prediction results
    10                     |   0.112
    1                      |   0.872
    23                     |   0.472
    ...

The ``result_info.json`` stores three informations:
  * ``format``: The format of the saved prediction results. By default, it is ``pytorch``.
  * ``emb_name``: A list of node types that have node prediction results saved. For example: ["ntype0", "ntype1"]
  * ``world_size``: The number of chunks (files) into which the prediction results of a particular node type are divided. For instance, if world_size is set to 8, there will be 8 files for each set of prediction results."


For edge prediction tasks, the prediction results are saved per edge type.
The sub-directory for an edge type is named as ``<src_ntype>_<relation_type>_<dst_ntype>``.
For instance, given an edge type ``("movie","rated-by","user")``, the corresponding
sub-directory is named as ``movie_rated-by_user``.
GraphStorm will also save an ``result_info.json`` file, which contains all
the metadata for the saved prediction results. The ``save_prediction_path``
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

    src_nids-00000.pt   |   dst_nids-00000.pt   |   predict.pt
                        |
    Source Node ID      |   Destination Node ID |   Prediction results
    10                  |   12                  |   0.112
    1                   |   20                  |   0.872
    23                  |   3                   |   0.472
    ...

The ``result_info.json`` stores three informations:
  * ``format``: The format of the saved prediction results. By default, it is ``pytorch``.
  * ``etypes``: A list of edge types that have edge prediction results saved. For example: [("movie","rated-by","user"), ("user","watched","movie")]
  * ``world_size``: The number of chunks (files) into which the prediction results of a particular edge type are divided. For instance, if world_size is set to 8, there will be 8 files for each set of prediction results."

**Note: The built-in GraphStorm inference pipeline
(launched by GraphStorm CLI) will process the saved prediction results
to convert the integer node ids into the raw node ids, which are usually string node ids.**
Details can be found in :ref:`GraphStorm Output Node ID Remapping<output-remapping>`
