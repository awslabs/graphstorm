.. _single-machine-training-inference:

Model Training and Inference on a Single Machine
-------------------------------------------------

While the :ref:`Standalone Mode Quick Start <quick-start-standalone>` tutorial introduces some basic concepts, commands, and steps of using GprahStorm CLIs in a single machine, this user guide provides more comprehensive description of the usage of GraphStorm CLIs in a single machine. In addition, the majority of the descriptions in this guide can be directly applied on :ref:`distributed clusters <distributed-cluster>`.

GraphStorm can support graph machine learning (GML) model training and inference for common GML tasks, including node classification, node regression, edge classification, edge regression, and link prediction. For each task, GraphStorm provide a dedicatd CLI for model training and inference. These CLIs share the same command template and some configurations, while each CLI has its unique task-specific configurations.

.. note:: 

    * Users can set CLI configurations either in CLI arguments or the configuration YAML file sepcified by the **-\-cf** argument. But values set in CLI arguments will overwrite the values set in the YAML file.
    * This guide only exlains some configurations commonly used. For the detailed explanations of GraphStorm CLI configurations, please refer to the :ref:` Model Training and Inference Configurations<configurations-run>`.

Node classification (NC) CLI for model training and inference
..............................................................

An example NC model training CLI is like the command below. 

.. code-block:: bash

    python -m graphstorm.run.gs_node_classification \
              --workspace /workspace_folder/ \
              --num-trainers 1 \
              --num-servers 1 \
              --part-config /tmp/ogbn_arxiv_nc_1p/ogbn-arxiv.json \
              --cf /graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml \
              --save-model-path /tmp/ogbn-arxiv-nc/models

