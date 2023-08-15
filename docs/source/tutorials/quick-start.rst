.. _quick-start-standalone:

Standalone Mode Quick Start Tutorial
=======================================
GraphStorm provides a set of tools, which can help users to use built-in datasets as examples to quickly learn the general steps of using GraphStorm.

GraphStorm is designed for easy-to-use GML models, particularly the graph neural network (GNN) models. Users only need to perform three operations:

- 1. Prepare Graph dataset in the required format as inputs of GraphStorm;
- 2. Launch GraphStorm training scripts and save the best models;
- 3. Launch GraphStorm inference scripts with saved models to predict.

This tutorial will use GraphStorm's built-in OGB-arxiv dataset for a node classification task to demonstrate these three steps.

.. warning::

    - All commands below are designed to run in a GraphStorm Docker container. Please refer to the :ref:`GraphStorm Docker environment setup<setup>` to prepare the Docker container environment. 

    - If you set up the :ref:`GraphStorm environment with pip Packages<setup_pip>`, please replace all occurrences of "2222" in the argument ``--ssh-port`` with **22**, and clone GraphStorm toolkits.

    - If use this method to setup GraphStorm environment, you may need to replace the ``python3`` command with ``python``, depending on your Python versions.

Download  and Partition OGB-arxiv Data
--------------------------------------
First run the below command.

.. code-block:: bash

    python3 /graphstorm/tools/partition_graph.py --dataset ogbn-arxiv \
                                                 --filepath /tmp/ogbn-arxiv-nc/ \
                                                 --num-parts 1 \
                                                 --output /tmp/ogbn_arxiv_nc_1p

This command will automatically download ogbn-arxiv graph data and split the graph into one partition. Outcomes of the command are a set of files saved in the ``/tmp/ogbn_arxiv_nc_1p/`` folder, as shown below.

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

.. _launch-training:

Launch Training
-----------------
GraphStorm currently relies on **ssh** to launch its scripts. Therefore before launch any scripts, users need to create an IP address file, which contains all private IP addresses in a cluster. If run GraphStorm in the Standalone mode, which run only in a **single machine**, as this tutorial does, users only need to run the following command to create an ``ip_list.txt`` file that has one row '**127.0.0.1**' as its content.

.. code-block:: bash

    touch /tmp/ogbn-arxiv-nc/ip_list.txt
    echo 127.0.0.1 > /tmp/ogbn-arxiv-nc/ip_list.txt

Then run the below command to start a training job that trains an built-in RGCN model to perform node classification on the OGB-arxiv.

.. code-block:: bash

    python3  -m graphstorm.run.gs_node_classification \
                --workspace /tmp/ogbn-arxiv-nc \
                --num-trainers 1 \
                --num-servers 1 \
                --num-samplers 0 \
                --part-config /tmp/ogbn_arxiv_nc_1p/ogbn-arxiv.json \
                --ip-config  /tmp/ogbn-arxiv-nc/ip_list.txt \
                --ssh-port 2222 \
                --cf /graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml \
                --save-model-path /tmp/ogbn-arxiv-nc/models

This command uses GraphStorm's training scripts and default settings defined in the `/graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml <https://github.com/awslabs/graphstorm/blob/main/training_scripts/gsgnn_np/arxiv_nc.yaml>`_ file. It will train an RGCN model by 10 epochs and save the model files after each epoch at the ``/tmp/ogbn-arxiv-nc/models`` folder whose contents are like the below structure.

.. code-block:: bash
    
    /tmp/ogbn-arxiv-nc/models
    |- epoch-0
        model.bin
        node_sparse_emb.bin
        optimizers.bin
    |- epoch-1
        ...
    |- epoch-9

In a single AWS g4dn.12xlarge instance, it takes around 8 seconds to finish one training and evaluation epoch by using **1 GPU**.

Launch inference
----------------
The output log of the training command also show which epoch achieves the best performance on the validation set. Users can use the saved model in this best performance epoch, e.g., epoch-7, to do inference.

The inference command is:

.. code-block:: bash

    python3 -m graphstorm.run.gs_node_classification \
               --inference \
               --workspace /tmp/ogbn-arxiv-nc \
               --num-trainers 1 \
               --num-servers 1 \
               --num-samplers 0 \
               --part-config /tmp/ogbn_arxiv_nc_1p/ogbn-arxiv.json \
               --ip-config  /tmp/ogbn-arxiv-nc/ip_list.txt \
               --ssh-port 2222 \
               --cf /graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml \
               --save-prediction-path /tmp/ogbn-arxiv-nc/predictions/ \
               --restore-model-path /tmp/ogbn-arxiv-nc/models/epoch-7/

This inference command predicts the classes of nodes in the testing set and saves the results, a Pytorch tensor file named "**predict-0.pt**", into the ``/tmp/ogbn-arxiv-nc/predictions/`` folder.

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
