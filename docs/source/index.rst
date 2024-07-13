
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

   graph-construction/index.rst

.. toctree::
   :maxdepth: 1
   :caption: Distributed Training
   :hidden:
   :glob:

   scale/distributed
   scale/sagemaker

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
   advanced/advanced-usages
   advanced/advanced-wholegraph
   advanced/multi-task-learning

GraphStorm is a graph machine learning (GML) framework designed for enterprise use cases. It simplifies the development, training and deployment of GML models on industry-scale graphs (measured in billons of nodes and edges) by providing scalable training and inference pipelines of GML models. GraphStorm comes with a collection of built-in GML models, allowing users to train a GML model with a single command, eliminating the need to write any code. Moreover, GraphStorm provides a wide range of configurations to customiz model implementations and training pipelines, enhancing model performance. In addition, GraphStorm offers a programming interface that enables users to train custom GML models in a distributed manner. Users can bring their own model implementations and leverage the GraphStorm training pipeline for scalability.

Getting Started
----------------

For beginners, please first start with the :ref:`Setup GraphStorm with pip Packages<setup_pip>`. This tutorial covers how to set up a GraphStorm Standalone mode environment for quick start.

Once successfully set up the GraphStorm running environment,

- follow the :ref:`GraphStorm Standalone Mode Quick-Start Tutorial<quick-start-standalone>` to use GraphStorm Command Line Interfaces (CLIs) to run examples based on GraphStorm built-in data and models, hence getting familiar with GraphStorm CLIs for training and inference.
- follow the :ref:`Use Your Own Graph Data Tutorial<use-own-data>` to prepare your own graph data for using GraphStorm CLIs.
- read the :ref:`GraphStorm Training and Inference Configurations<configurations-run>` to learn the various configurations provided by GraphStorm for CLIs that can help to achieve the best performance.

Scale to Giant Graphs
----------------------

For users who wish to train and run infernece on very large graphs,

- follow the :ref:`Setup GraphStorm Docker Environment<setup_docker>` tutorial to create GraphStorm dockers for distributed runtime environments.
- follow the :ref:`Use GraphStorm Distributed Data Processing<gs-processing>` tutorial to process and construction large graphs in the Distributed mode.
- follow the :ref:`Use GraphStorm in a Distributed Cluster<distributed-cluster>` tutorial to use GraphStorm in the Distributed mode.
- follow the :ref:`Use GraphStorm on SageMaker<distributed-sagemaker>` tutorial to use GraphStorm in the Distribute mode based on Amazon SageMaker.

Use GraphStorm APIs
---------------------

For users who wish to customize GraphStorm for their specific needs, follow the :ref:`GraphStorm API Programming Example Notebooks<programming-examples>` to explore GraphStorm APIs, learn how to use GraphStorm APIs to reproduce CLIs pipelines, and then customize GraphStorm components for specific requirements. Users can find the details of GraphStorm APIs in the :ref:`API Reference<api-reference>` documentations.

Advanced Topics
----------------

- For users who want to use their own GML models in GraphStorm, follow the :ref:`Use Your Own GNN Models<use-own-models>` tutorial to learn the programming interfaces and the steps of how to modify users' own models.
- For users who want to leverage language models on nodes with text features, follow the :ref:`Use Language Model in GraphStorm<language_models>` tutorial to learn how to leverage BERT models to use text as node features in GraphStorm.
- There are various usages of GraphStorm to both speed up training process and help to boost model performance for link prediction tasks. Users can find these usages in the :ref:`Link Prediction Learning in GraphStorm<_link_prediction_usage>` page.
- GraphStorm team has been working with NVIDIA team to integrate the NVIDIA's WholeGraph library into GraphStorm for speed-up of feature copy. Users can follow the :ref:`Use WholeGraph in GraphStorm<advanced_wholegraph>` tutorial to know more details.
- In v0.3, GraphStorm releases an experimental feature to support multi-task learning on the same graph, allowing users to define multiple training targets on different nodes and edges within a single training loop. Users can check the :ref:`Multi-task Learning in GraphStorm<multi_task_learning>` tutorial to know more details.

Contribution
-------------
GraphStorm is free software; you can redistribute it and/or modify it under the terms of the Apache License 2.0. We welcome contributions.
Join us on `GitHub <https://github.com/awslabs/graphstorm>`_.
