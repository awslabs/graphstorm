.. _single-machine-training-inference:

Model Training and Inference on a Single Machine
-------------------------------------------------
While the :ref:`Standalone Mode Quick Start <quick-start-standalone>` tutorial introduces some basic concepts, commands, and steps of using GprahStorm CLIs in a single machine, this user guide provides more comprehensive description of the usage of GraphStorm CLIs in a single machine. In addition, the majority of the descriptions in this guide can be directly applied on :ref:`distributed clusters <distributed-cluster>`.

GraphStorm can support graph machine learning (GML) model training and inference for common GML tasks, including node classification, node regression, edge classification, edge regression, and link prediction. For each task, GraphStorm provide a dedicatd CLI for model training and inference. These CLIs share the same command template and some configurations, while each CLI has its unique task-specific configurations.

Task-specific CLI template for model training and inference
............................................................
GraphStorm model training and inference CLIs like the commands below. 

.. code-block:: bash

    # Model training
    python -m graphstorm.run.TASK_COMMAND \
              --num-trainers 1 \
              --part-config data.json \
              --cf config.yaml \
              --save-model-path model_path/

    # Model inference
    python -m graphstorm.run.TASK_COMMAND \
              --inference \
              --num-trainers 1 \
              --part-config data.json \
              --cf config.yaml \
              --restore-model-path model_path/ \
              --save-prediction-path pred_path/

In the above two CLIs, the ``TASK_COMMAND`` represents one of the five task-specific commands:

    * ``gs_node_classification`` for node classification tasks;
    * ``gs_node_regression`` for node regression tasks;
    * ``gs_edge_classification`` for edge classification tasks;
    * ``gs_edge_regression`` for edge regression tasks;
    * ``gs_link_prediction`` for link prediction tasks.

These task-specific commands work for both model training and inference except that inference CLI needs to add the **-\-inference** argument to indicate this is an inference CLI, and the **-\-restore-model-path** argument that indicate the path of the saved model checkpoint.

For a single machine the argument **-\--num-trainers** can configure how many GPUs or CPU processes to be used. If using a GPU installed machine, the value of **-\--num-trainers** should be **equal or less** than the total number of available GPUs, while in a CPU-only machine, the value could be less than the total number of CPU processes to avoid unexpected errors.

The model training and inference CLIs use the **-\--part-config** argument to specify the partitioned graph data by giving the path of the `*.json` file generated from the :ref:`GraphStorm Graph Construction <graph_construction>`.

While the CLIs could be very simple as the template demonstrated, users can use a YAML file to set various configurations that could make full use of the rich functions and features provided by GraphStorm. The YAML file will be specified to the **-\-cf** argument. GraphStorm has a set of `example YAML files <https://github.com/awslabs/graphstorm/tree/main/training_scripts>`_ available for reference.

.. note:: 

    * Users can set CLI configurations either in CLI arguments or the configuration YAML file specified by the **-\-cf** argument. But values set in CLI arguments will overwrite the values of the same configurations set in the YAML file.
    * This guide only explains some configurations commonly used. For the detailed explanations of GraphStorm CLI configurations, please refer to the :ref:` Model Training and Inference Configurations<configurations-run>`.

Task-agnostic CLI for model training and inference
...................................................
While task-specific CLIs allow users to quickly perform GrphStorm built-in GNN models on the tasks supported, users may build their own GNN models like the :ref:`Use Your Own Models <use-own-models>` tutorial. For these customized models, users can use the task-agnostic CLI like the following template.

.. code-block:: bash

    # Model training
    python -m graphstorm.run.launch \
              --num-trainers 1 \
              --part-config data.json \
              --save-model-path model_path/ \
              customized_model.py customized_arguments

    # Model inference
    python -m graphstorm.run.launch \
              --inference \
              --num-trainers 1 \
              --part-config data.json \
              --restore-model-path model_path/ \
              --save-prediction-path pred_path/
              customized_model.py customized_arguments

The task-agnostic CLI command (``launch``) has similar tempalte as the task-specific CLIs except that it takes the customized model stored as a ``.py`` file as an argument. And in case the customized model has its own arguments, they should be placed after the customized model python file.
