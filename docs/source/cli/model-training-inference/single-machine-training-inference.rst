.. _single-machine-training-inference:

Model Training and Inference on a Single Machine
-------------------------------------------------

While the :ref:`Standalone Mode Quick Start <quick-start-standalone>` tutorial introduces some basic concepts, commands, and steps of using GprahStorm CLIs in a single machine, this user guide provides more comprehensive description of the usage of GraphStorm CLIs in a single machine. In addition, the majority of the descriptions in this guide can be directly applied on :ref:`distributed clusters <distributed-cluster>`.

GraphStorm can support graph machine learning (GML) model training and inference for common GML tasks, including node classification, node regression, edge classification, edge regression, and link prediction. For each task, GraphStorm provide a dedicatd CLI for model training and inference. These CLIs share the same command template and some configurations, while each CLI has its unique task-specific configurations.

.. note:: 

    * Users can set CLI configurations either in CLI arguments or the configuration YAML file sepcified by the **-\-cf** argument. But values set in CLI arguments will overwrite the values set in the YAML file.
    * This guide only exlains some configurations commonly used. For the detailed explanations of GraphStorm CLI configurations, please refer to the :ref:` Model Training and Inference Configurations<configurations-run>`.

Common CLI template for model training and inference
.......................................................

A GraphStorm model training and inference CLI is like the commands below. 

.. code-block:: bash

    # Model training
    python -m graphstorm.run.TASK_COMMAND \
              --workspace workspace_folder/ \
              --num-trainers 1 \
              --part-config data.json \
              --cf config.yaml \
              --save-model-path model_path/

    # Model inference
    python -m graphstorm.run.TASK_COMMAND \
          --inference \
          --workspace workspace_folder/ \
          --num-trainers 1 \
          --part-config data.json \
          --cf config.yaml \
          --restore-model-path model_path/ \
          --save-prediction-path pred_path/


