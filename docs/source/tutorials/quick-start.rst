.. _quick-start:

Quick Start Tutorial
====================
GraphStorm provides a set of tools, which can help users to use built-in datasets as examples to quickly learn how to use GraphStorm.

GraphStorm is designed for easy-to-use graph neural networks (GNNs). Users only need to perform three operations:

- 1. Prepare Graph dataset;
- 2. Launch training scripts;
- 3. Launch inference scripts.

This tutorial will use GraphStorm's built-in OGB-arxiv dataset for a node classification task to demonstrate these three steps.

.. note::

    All commands below run in a GraphStorm Docker container. Please refer to the :ref:`GraphStorm Docker environment setup<setup>` to prepare your environment.

Download  and Partition OGB-arxiv Data
--------------------------------------

.. code-block:: bash

    python3 /graphstorm/tools/partition_graph.py --dataset ogbn-arxiv \
                                             --filepath /tmp/ogbn-arxiv-nc/ \
                                             --num_parts 1 \
                                             --output /tmp/ogbn_arxiv_nc_1p

This command will automatically download ogbn-arxiv data and split the graph into one partition. Outcomes of the command are a set of files saved in the /tmp/ogbn_arxiv_nc_1p/ folder, as shown below.

.. code-block:: bash

    /tmp/ogbn_arxiv_nc_1p:
    ogbn-arxiv.json
    |- part0:
        edge_feat.dgl
        graph.dgl
        node_feat.dgl

The ``ogbn-arxiv.json`` file contains meta data about the built distributed DGL graph. As the partition command specifies to create one partition, there is one sub-folder, named ``part0``.  Files in the sub-folder includes three types of data, i.e., the graph structure, the node features, and edge features.

.. _launch-training:

Launch Training
-----------------
GraphStorm currently relies on **ssh** to launch its scripts. Therefore before launch any scripts, users need to create an IP address file, which contains all private IP addresses in a cluster. If run GraphStorm in a signle machine, as this tutorial does, only need to run the following command to create an ``ip_list.txt`` file with one row '**127.0.0.1**' as its content.

.. code-block:: bash

    touch /tmp/ogbn-arxiv-nc/ip_list.txt
    echo 127.0.0.1 >/tmp/ogbn-arxiv-nc/ip_list.txt

Then run below command to start a training job that train an built-in RGCN model to perform node classification on OGB-arxiv.

.. code-block:: bash

    python3  -m graphstorm.run.gs_node_classification \
             --workspace /tmp/ogbn-arxiv-nc \
             --num_trainers 1 \
             --num_servers 1 \
             --num_samplers 0 \
             --part_config /tmp/ogbn_arxiv_nc_1p/ogbn-arxiv.json \
             --ip_config  /tmp/ogbn-arxiv-nc/ip_list.txt \
             --ssh_port 2222 \
             --cf /graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml \
             --save-model-path /tmp/ogbn-arxiv-nc/models

This command uses GraphStorm's training scripts and default settings defined in the `/graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml <https://github.com/awslabs/graphstorm/blob/main/training_scripts/gsgnn_np/arxiv_nc.yaml>`_. It will train an RGCN model by 10 epochs and save the model files after epoch at the ``/tmp/ogbn-arxiv-nc/models`` folder as the shown structure.

.. code-block:: bash
    
    /tmp/ogbn-arxiv-nc/models
    |- epoch-0
        model.bin
        node_sparse_emb.bin
        optimizers.bin
    |- epoch-1
        ...
    |- epoch-9

In an AWS g4dn.12xlarge instance, it takes around 8 seconds to finish one training and evaluation epoch by using 1 GPU.

Launch inference
----------------
The output log of the training command also show which epoch achieve the best performance on the validation set. Users can use saved model in this epoch, e.g., epoch-7, to perform inference as the follow command.

.. code-block:: bash

    python3 -m graphstorm.run.gs_node_classification \
               --inference \
               --workspace /tmp/ogbn-arxiv-nc \
               --num_trainers 1 \
               --num_servers 1 \
               --num_samplers 0 \
               --part_config /tmp/ogbn_arxiv_nc_1p/ogbn-arxiv.json \
               --ip_config  /tmp/ogbn-arxiv-nc/ip_list.txt \
               --ssh_port 2222 \
               --cf /graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml \
               --save-prediction-path /tmp/ogbn-arxiv-nc/predictions/ \
               --restore-model-path /tmp/ogbn-arxiv-nc/models/epoch-7/

This inference command predicts the classes of nodes in the testing set and saves the results, a Pytorch tensor file named "**predict-0.pt**", into the ``/tmp/ogbn-arxiv-nc/predictions/`` folder.

**That is it!** You have learnt how to use GraphStorm in three steps. 

Next users can check the :ref:`Use Your Own Graph Data<use-own-data>` guide to prepare your own graph data for using GraphStorm.

Clean Up
----------
Once finish graph machine learning tasks, users can exit the GraphStorm Docker container with command exit and then stop the container to restore computation resources.

Run this command in the **container running environment** to leave GraphStorm container.

.. code-block:: bash

    exit

Run this command in the **instance environment** to stop the GprahStorm Docker container.

.. code-block:: bash

    docker stop test

Make sure you give the correct container name in this above command. Here it stops the container name ``test``.

Then users can use this command to check the status of all Docker containers. The container with name ``test`` should have a "**STATUS**" like "**Exited (0) ** ago**".

.. code-block::

    docker ps -a
