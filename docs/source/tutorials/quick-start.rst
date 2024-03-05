.. _quick-start-standalone:

Standalone Mode Quick Start
============================
GraphStorm provides a set of tools, which can help users to use built-in datasets as examples to quickly learn the general steps of using GraphStorm.

GraphStorm is designed for easy-to-use GML models, particularly the graph neural network (GNN) models. Users only need to perform three operations:

- 1. Prepare Graph dataset in the required format as inputs of GraphStorm;
- 2. Launch GraphStorm training scripts and save the best models;
- 3. Launch GraphStorm inference scripts with saved models to predict the test set or generate node embeddings.

This tutorial will use GraphStorm's built-in OGB-arxiv dataset for a node classification task to demonstrate these three steps in GraphStorm's Standalone mode, i.e., running GraphStorm scripts in one instance with either CPUs or GPUs.

In terms of the Standalone mode, users can use the :ref:`Setup GraphStorm with pip Packages<setup_pip>` method to install GraphStorm in an instance.

Download and Partition OGB-arxiv Data
--------------------------------------
First run the below command.

.. code-block:: bash

    python /graphstorm/tools/partition_graph.py --dataset ogbn-arxiv \
                                                --filepath /tmp/ogbn-arxiv-nc/ \
                                                --num-parts 1 \
                                                --output /tmp/ogbn_arxiv_nc_1p

This command will automatically download the ogbn-arxiv graph data and split the graph into one partition for node classification. Outcomes of the command are a set of files saved in the ``/tmp/ogbn_arxiv_nc_1p/`` folder, as shown below.

.. code-block:: bash

    /tmp/ogbn_arxiv_nc_1p:
    ogbn-arxiv.json
    node_mapping.pt
    edge_mapping.pt
    |- part0:
        edge_feat.dgl
        graph.dgl
        node_feat.dgl

The ``ogbn-arxiv.json`` file contains meta data about the built distributed DGL graph. Because the command specifies to create one partition with the argument ``--num-parts 1``, there is one sub-folder, named ``part0``.  Files in the sub-folder includes three types of data, i.e., the graph structure (``graph.dgl``), the node features (``node_feat.dgl``), and edge features (``edge_feat.dgl``). The ``node_mapping.pt`` and ``edge_mapping.pt`` contain the ID mapping between the raw node and edge IDs with the built graph's node and edge IDs.

Running the following command can download the ogbn-arxiv graph data and split the graph into one partition for a link prediction task. And the output of the command is same as the above folder structure, except that the graph is split on edges.

.. code-block:: bash

    python /graphstorm/tools/partition_graph_lp.py --dataset ogbn-arxiv \
                                                   --filepath /tmp/ogbn-arxiv-lp/ \
                                                   --num-parts 1 \
                                                   --output /tmp/ogbn_arxiv_lp_1p/

.. _launch-training:

Launch Training
-----------------

Run the below command to start a training job that trains a built-in RGCN model to perform node classification on the OGB-arxiv.

.. code-block:: bash

    python -m graphstorm.run.gs_node_classification \
              --workspace /tmp/ogbn-arxiv-nc \
              --num-trainers 1 \
              --num-servers 1 \
              --part-config /tmp/ogbn_arxiv_nc_1p/ogbn-arxiv.json \
              --cf /graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml \
              --save-model-path /tmp/ogbn-arxiv-nc/models

This command uses GraphStorm's training scripts and default settings defined in the `/graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml <https://github.com/awslabs/graphstorm/blob/main/training_scripts/gsgnn_np/arxiv_nc.yaml>`_ file. It will train an RGCN model by 10 epochs and save the model files after each epoch at the ``/tmp/ogbn-arxiv-nc/models`` folder whose contents are like the below structure.

.. code-block:: bash

    /tmp/ogbn-arxiv-nc/models
    |- epoch-0
        model.bin
        |- node
            sparse_emb_00000.pt
        optimizers.bin
    |- epoch-1
        ...
    |- epoch-n

In terms of link prediciton, run the following command will train an RGCN model with the `/graphstorm/training_scripts/gsgnn_lp/arxiv_lp.yaml <https://github.com/awslabs/graphstorm/blob/main/training_scripts/gsgnn_lp/arxiv_lp.yaml>`_ file.

.. code-block:: bash

    python -m graphstorm.run.gs_link_prediction \
              --workspace /tmp/ogbn-arxiv-lp \
              --num-trainers 1 \
              --num-servers 1 \
              --part-config /tmp/ogbn_arxiv_lp_1p/ogbn-arxiv.json \
              --cf /graphstorm/training_scripts/gsgnn_lp/arxiv_lp.yaml \
              --save-model-path /tmp/ogbn-arxiv-lp/models

Launch inference
----------------
The output log of the training command also show which epoch achieves the best performance on the validation set, like in the below snipet.

.. code-block:: yaml

    INFO:root:best_test_score: {'accuracy': 0.6055593276135218}
    INFO:root:best_val_score: {'accuracy': 0.6330078190543307}
    INFO:root:peak_GPU_mem_alloc_MB: 370.83056640625
    INFO:root:peak_RAM_mem_alloc_MB: 3985.765625
    INFO:root:best validation iteration: 356
    INFO:root:best model path: /tmp/ogbn-arxiv-nc/models/epoch-7

Users can use the saved model in this best performance epoch, e.g., epoch-7, to do inference.

The inference command is:

.. code-block:: bash

    python -m graphstorm.run.gs_node_classification \
              --inference \
              --workspace /tmp/ogbn-arxiv-nc \
              --num-trainers 1 \
              --num-servers 1 \
              --part-config /tmp/ogbn_arxiv_nc_1p/ogbn-arxiv.json \
              --cf /graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml \
              --save-prediction-path /tmp/ogbn-arxiv-nc/predictions/ \
              --restore-model-path /tmp/ogbn-arxiv-nc/models/epoch-7/

This inference command predicts the classes of nodes in the testing set and saves the results, a list of parquet files named **predict-00000_00000.parquet**, **predict-00001_00000.parquet**, ..., into the ``/tmp/ogbn-arxiv-nc/predictions/node/`` folder. Each parquet file has two columns, `nid` column for storing node IDs and `pred` column for storing prediction results.

Inference on link prediction is similar as shown in the command below.

.. code-block:: bash

    python3 -m graphstorm.run.gs_link_prediction \
               --inference \
               --workspace /tmp/ogbn-arxiv-lp \
               --num-trainers 1 \
               --num-servers 1 \
               --part-config /tmp/ogbn_arxiv_lp_1p/ogbn-arxiv.json \
               --cf /graphstorm/training_scripts/gsgnn_lp/arxiv_lp.yaml \
               --save-embed-path /tmp/ogbn-arxiv-lp/predictions/ \
               --restore-model-path /tmp/ogbn-arxiv-lp/models/epoch-2/

The inference outputs the saved embeddings, a list of parquet files named **embed-00000_00000.parquet**, **embed-00001_00000.parquet**, ...,  in the ``/tmp/ogbn-arxiv-lp/predictions/node/`` folder. Each parquet file has two columns, `nid` column for storing node IDs and `emb` column for storing embeddings.

Generating Embedding
--------------------
If users only need to generate node embeddings instead of doing predictions on the graph, users can use saved model and the same yaml configuration file used in training to achieve that with the ``gs_gen_node_embedding`` command:

.. code-block:: bash

    python -m graphstorm.run.gs_gen_node_embedding \
              --workspace /tmp/ogbn-arxiv-nc \
              --num-trainers 1 \
              --part-config /tmp/ogbn_arxiv_nc_1p/ogbn-arxiv.json \
              --cf /graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml \
              --save-embed-path /tmp/ogbn-arxiv-nc/saved_embed \
              --restore-model-path /tmp/ogbn-arxiv-nc/models/epoch-7/ \
              --use-mini-batch-infer true

Users need to specify ``--restore-model-path`` and ``--save-embed-path`` when using the command above to generate node embeddings, and the node embeddings will be saved into the folder specified by the ``--save-embed-path`` argument. Outputs of the above command is like:

.. code-block:: bash

    /tmp/ogbn-arxiv-nc/saved_embed
        emb_info.json
        node/
            node_embed-00000.pt


For node classification/regression task, ``target_ntype`` is necessary, the command will generate and save node embeddings on ``target_ntype``. If it requires generating embeddings on multiple nodes, the input ``target_ntype`` should be a list of node types.

For edge classification/regression task, ``target_etype`` is necessary, the command will generate and save node embeddings on source and destination node types defined in the ``target_etype``. If it requires generating embeddings on multiple nodes, the input ``target_etype`` should be a list of edge types.

For link prediction task, it will generate and save node embeddings for all node types.

The saved result will be like:

.. code-block:: bash

    /tmp/saved_embed
        emb_info.json
        node_type1/
            embed-00000_00000.parquet
            embed-00000_00001.parquet
            ...
        node_type2/
            embed-00000_00000.parquet
            embed-00000_00001.parquet
            ...

**That is it!** You have learnt how to use GraphStorm in three steps.

Next users can check the :ref:`Use Your Own Graph Data<use-own-data>` tutorial to prepare your own graph data for using GraphStorm.

Clean Up
----------
Once finished with GML tasks, users can exit the GraphStorm Docker container with command ``exit`` and then stop the container to restore computation resources.

Run this command in the **container running environment** to leave the GraphStorm container.

.. code-block:: bash

    exit

Run this command in the **instance environment** to stop the GprahStorm Docker container.

.. code-block:: bash

    docker stop test

Make sure you give the correct container name in the above command. Here it stops the container named ``test``.

Then users can use this command to check the status of all Docker containers. The container with the name ``test`` should have a "**STATUS**" like "**Exited (0) ** ago**".

.. code-block::

    docker ps -a
