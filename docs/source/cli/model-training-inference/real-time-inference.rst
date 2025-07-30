.. _real-time-inference-on-sagemaker:

Real-time Inference on Amazon SageMaker
----------------------------------------
GraphStorm CLIs for model inference on :ref:`signle machine <single-machine-training-inference>`,
:ref:`distributed clusters <distributed-cluster>`, and :ref:`Amazon SageMaker <distributed-sagemaker>`
are designed to handle enterprise-level data, which could take minutes to hours for predicting a large
number of target nodes/edges or generating embeddings for all nodes.

In certain cases when you want to predict a few targets only and hope to obtain results in short time, e.g., less than one second, you will need a 7*24 running server to host trained model and response to
inference request in real time. GraphStorm provides a CLI to deploy a trained model as a SageMaker
real-time inference endpoint. To invoke this endpoint, you will need to extract a subgraph around a few
target nodes/edges and convert the subgraph and associated features into a JSON object as payload of
an invoke request.

Inductive versus Deductive Inference
.....................................
One caveat about GraphStorm real-time is the inference mode. Models in transductive inference mode can
not make predictions for newly appeared nodes and edges, whereas in inductive mode, models can handle
new nodes and edges. A demonstration of the difference between transductive and inductive mode is shown
in the following figure.

.. figure:: ../../../tutorial/inductive-deductive.jp
    :align: center

While the :ref:`Standalone Mode Quick Start <quick-start-standalone>` tutorial introduces some basic concepts, commands, and steps of using GprahStorm CLIs on a single machine, this user guide provides more detailed description of the usage of GraphStorm CLIs in a single machine. In addition, the majority of the descriptions in this guide can be directly applied to :ref:`model training and inference on distributed clusters <distributed-cluster>`.

GraphStorm can support graph machine learning (GML) model training and inference for common GML tasks, including **node classification**, **node regression**, **edge classification**, **edge regression**, and **link prediction**. Since the :ref:`multi-task learning <multi_task_learning>` feature released in v0.3 is in experimental stage, formal documentations about this feature will be released later when it is mature.

For each task, GraphStorm provide a dedicated CLI for model training and inference. These CLIs share the same command template and some configurations, while each CLI has its unique task-specific configurations. GraphStorm also has a task-agnostic CLI for users to run your customized models.

While the CLIs could be very simple as the template demonstrated, users can leverage a YAML file to set a variaty of GraphStorm configurations that could make full use of the rich functions and features provided by GraphStorm. The YAML file will be specified to the **-\-cf** argument. GraphStorm has a set of `example YAML files <https://github.com/awslabs/graphstorm/tree/main/training_scripts>`_ available for reference.

.. note:: 

    * Users can set CLI configurations either in CLI arguments or the configuration YAML file. But values set in CLI arguments will overwrite the values of the same configuration set in the YAML file.
    * This guide only explains a few commonly used configurations. For the detailed explanations of GraphStorm CLI configurations, please refer to the :ref:`Model Training and Inference Configurations <configurations-run>` section.

Task-agnostic CLI for model training and inference
...................................................
While task-specific CLIs allow users to quickly perform GML tasks supported by GraphStorm, users may build their own GNN models as described in the :ref:`Use Your Own Models <use-own-models>` tutorial. To put these customized models into GraphStorm model training and inference pipeline, users can use the task-agnostic CLI as shown in the examples below.


.. code-block:: bash

    # Model training
    python -m graphstorm.run.launch \
              --num-trainers 1 \
              --part-config data.json \
              customized_model.py --save-model-path model_path/ \
                                  customized_arguments 

    # Model inference
    python -m graphstorm.run.launch \
              --inference \
              --num-trainers 1 \
              --part-config data.json \
              customized_model.py --restore-model-path model_path/ \
                                  --save-prediction-path pred_path/ \
                                  customized_arguments

The task-agnostic CLI command (``launch``) has similar tempalte as the task-specific CLIs except that it takes the customized model, which is stored as a ``.py`` file, as an argument. And in case the customized model has its own arguments, they should be placed after the customized model python file.
