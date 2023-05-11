
Welcome to GraphStorm Documentation and Tutorials
==================================================

.. toctree::
   :maxdepth: 2
   :caption: Get Started
   :hidden:
   :glob:

   install/env-setup
   tutorials/quick-start
   tutorials/own-data
   configuration/index

.. toctree::
   :maxdepth: 1
   :caption: Scale to Giant Graphs
   :hidden:
   :glob:

   scale/distributed

.. toctree::
   :maxdepth: 1
   :caption: Advanced Topics
   :hidden:
   :glob:

   advanced/own-models

.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   :hidden:
   :glob:

GraphStorm is a graph machine learning (GML) framework for enterprise use cases. It simplifies the development, training and deployment of GML models for industry-scale graphs
by providing scalable training and inference pipelines of GML models for extremely large graphs (measured in billons of nodes and edges).
GraphStorm provides a collection of built-in GML models and users can train a GML model with a single command without writing any code. To help develop state-of-the-art models,
GraphStorm provides a large collection of configurations for customizing model implementations and training pipelines to improve model performance. GraphStorm also provides a programming
interface to train any custom GML model in a distributed manner. Users provide their own model implementations and use GraphStorm training pipeline to scale.

Getting Started
----------------

For absolute beginners, please first start with the :ref:`GraphStorm Docker environment setup<setup>`. It will cover the topic of how to set up a Docker environment and build a GraphStorm Docker image, which is the Standalone running environment of GraphStorm. We are working on supporting more running environments for GraphStorm.

Once the GraphStorm Docker running environment is ready, 

- follow the :ref:`GraphStorm Quick-Start Guide<quick-start>` guide to run examples using GraphStorm built-in data and models to get familiar with GraphStorm's usage of training and inference.
- follow the :ref:`Use Your Own Graph Data<use-own-data>` guide to prepare your own graph data for GraphStorm.
- read the :ref:`GraphStorm Training and Inference Configurations<configurations-run>` to learn the various configurations provided by GraphStorm that can help to achieve the best performance.

Scale to Giant Graphs
---------------------------------

For experienced users who wish to train and run infernece on very large graphs,

- follow the :ref:`Use GraphStorm in a Distributed Cluster<distributed-cluster>` tutorial to use GraphStorm in Distributed mode.

Avanced Topics
--------------------

For users who want to use their own GML models in GraphStorm, 

- follow the :ref:`Use Your Own GNN Models<use-own-models>` tutorial to learn the programming interfaces and the general methods.
- [WIP] GraphStorm APIs

Contribution
-------------
GraphStorm is free software; you can redistribute it and/or modify it under the terms of the Apache License 2.0. We welcome contributions.
Join us on `GitHub <https://github.com/awslabs/graphstorm>`_.
