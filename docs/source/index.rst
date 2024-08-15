
Welcome to the GraphStorm Documentation and Tutorials
=======================================================

.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :hidden:
   :glob:

   install/env-setup
   tutorials/quick-start
   tutorials/own-data
   configuration/index

.. toctree::
   :maxdepth: 3
   :caption: Command Line Interface User Guide
   :hidden:
   :glob:

   cli/graph-construction/index.rst
   cli/model-training-inference/index.rst

.. toctree::
   :maxdepth: 2
   :caption: Programming Interface User Guide
   :hidden:
   :titlesonly:
   :glob:

   api/notebooks/index
   api/references/index

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics
   :hidden:
   :glob:

   advanced/own-models
   advanced/language-models
   advanced/link-prediction
   advanced/advanced-wholegraph
   advanced/multi-task-learning
   advanced/advanced-usages
   single-machine-gconstruct

GraphStorm is a graph machine learning (GML) framework designed for enterprise use cases. It simplifies the development, training and deployment of GML models on industry-scale graphs (measured in billons of nodes and edges) by providing scalable training and inference pipelines of GML models. GraphStorm comes with a collection of built-in GML models, allowing users to train a GML model with a single command, eliminating the need to write any code. Moreover, GraphStorm provides a wide range of configurations to customiz model implementations and training pipelines, enhancing model performance. In addition, GraphStorm offers a programming interface that enables users to train custom GML models in a distributed manner. Users can bring their own model implementations and leverage the GraphStorm training pipeline for scalability.

Getting Started
----------------

For beginners, please first start with the :ref:`Setup GraphStorm with pip Packages<setup_pip>`. This tutorial covers how to set up a GraphStorm Standalone mode environment for quick start.

Once successfully set up the GraphStorm running environment,

- follow the :ref:`GraphStorm Standalone Mode Quick-Start Tutorial<quick-start-standalone>` to use GraphStorm Command Line Interfaces (CLIs) to run examples based on GraphStorm built-in data and models, hence getting familiar with GraphStorm CLIs for training and inference.
- follow the :ref:`Use Your Own Graph Data Tutorial<use-own-data>` to prepare your own graph data for using GraphStorm model training and inference pipelines or APIs.
- read the :ref:`Model Training and Inference Configurations<configurations-run>` to learn the various configurations provided by GraphStorm for CLIs that can help to achieve the best performance.

GraphStorm provides two types of interfaces, i.e., Command Line Interfaces (CLIs) and Application Programming Interfaces (APIs), for users to conduct GML tasks for different purposes.

The CLIs abstract away the complexity of the GML pipeline for users to quickly build, train, and deploy models using common recipes. Meanwhile, the APIs reveal the major components by which GraphStorm constructs the GML pipelines. Users can levearge these APIs to customize GraphStorm for their specific needs.

GraphStorm CLIs User Guide
---------------------------

GraphStorm CLIs include two major functions, i.e., Graph Construction, and Model Training and Inference.

The :ref:`GraphStorm Graph Construction<graph_construction>` documentations explain how to construct distributed DGL graphs that can be use in GraphStorm training and inference pipelines. For relatively small data, users can :ref:`construct graphs on a single machine<single-machine-gconstruction>`. When dealing with very large data that can not be fit into memory of a single machine, users can refer to the :ref:`distributed graph construction <distributed-gconstruction>` documentations, knowing how to set up distributed environments and construct graphs using different infrastructures.

While the :ref:`GraphStorm Standalone Mode Quick-Start Tutorial<quick-start-standalone>` provides some information of using GraphStorm CLIs on a single machine, the :ref:`Model Training and Inference on a Single Machine <single-machine-training-inference>` documentation provides more detailed guidance. 

Similar as the documentations of distributed graph construction, the distributed model training and inference user guide explains how to set up distributed environments and run GraphStorm model training and inference using a :ref:`Distributed Cluster <distributed-cluster>` or :ref:`Amazon SageMaker <distributed-sagemaker>` to deal with enterprise-level graphs.

GraphStorm APIs User Guide
---------------------------

The released GraphStorm APIs list the major components that can help users to develop GraphStorm-like GML pipelines, or customize components such as GNN models, training conctrolers for their specific needs.

To help users use these APIs, GraphStorm also released a set of Jupyter notebooks at :ref:`GraphStorm API Programming Example Notebooks<programming-examples>`. By running these notebooks, users can explore some APIs, learn how to use APIs to reproduce CLIs pipelines, and then customize GraphStorm components for specific requirements. 

Users can find the comprehensive descriptions of these GraphStorm APIs in the :ref:`API Reference<api-reference>` documentations. For unrelease APIs, we encourage users to read their source code. If users want to have more APIs formally released, please raise issues at the `GraphStorm GitHub Repository <https://github.com/awslabs/graphstorm/issues>`_.

Advanced Topics
----------------

- For users who want to use their own GML models in GraphStorm, follow the :ref:`Use Your Own GNN Models<use-own-models>` tutorial to learn the programming interfaces and the steps of how to modify users' own models.
- For users who want to leverage language models on nodes with text features, follow the :ref:`Use Language Model in GraphStorm<language_models>` tutorial to learn how to leverage BERT models to use text as node features in GraphStorm.
- There are various usages of GraphStorm to both speed up training process and help to boost model performance for link prediction tasks. Users can find these usages in the :ref:`Link Prediction Learning in GraphStorm<link_prediction_usage>` page.
- GraphStorm team has been working with NVIDIA team to integrate the NVIDIA's WholeGraph library into GraphStorm for speed-up of feature copy. Users can follow the :ref:`Use WholeGraph in GraphStorm<advanced_wholegraph>` tutorial to know more details.
- In v0.3, GraphStorm releases an experimental feature to support multi-task learning on the same graph, allowing users to define multiple training targets on different nodes and edges within a single training loop. Users can check the :ref:`Multi-task Learning in GraphStorm<multi_task_learning>` tutorial to know more details.

Contribution
-------------
GraphStorm is free software; you can redistribute it and/or modify it under the terms of the Apache License 2.0. We welcome contributions.
Join us on `GitHub <https://github.com/awslabs/graphstorm>`_.
