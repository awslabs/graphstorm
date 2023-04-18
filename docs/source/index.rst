.. graphstorm documentation master file, created by
   sphinx-quickstart on Sat Apr 15 22:37:29 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GraphStorm Documentation and Tutorials
==================================================

.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :hidden:
   :glob:

   install/index
   tutorials/index
   tutorials/data-own
   configuration/index

.. toctree::
   :maxdepth: 2
   :caption: Advanced Materials:
   :hidden:
   :glob:

.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   :hidden:
   :glob:

GraphStorm, or GraphStorm Framework (GSF), is the next generation Graph Machine Learning (GML) framework for enterprise use cases, e.g. Search, Recommendation, CTR (Click Throughput Rate) boosting, etc. GraphStorm simplifies the development, training and deployment of GML models for industry-scale graphs that contain hundreds of billions of edges by providing easy-to-use and scalable graph data processing, training and inference pipelines for various GML models, particularly Graph Neural Network(GNN) models, and downstream tasks. 

Getting Started
---------------

For absolute beginners, please first start with the :ref:`GraphStorm Docker environment setup<setup>`. It will cover the topic of how to set up a Docker environment and build a GraphStorm Docker image, which is the Standalone running environment of GraphStorm. We are working on setting more running environments for GraphStorm.

Once the GraphStorm Docker running environment is ready, 

- follow the :ref:`GraphStorm Quick-Start Guide<quick-start>` to run GraphStorm built-in data and models to get familiar with GraphStorm's usage of training and inference.
- follow the :ref:`Prepare Your Own Graph Data<use-own-data>` to use your own graph data in GraphStorm.
- read the :ref:`GraphStorm Configuration<configurations>` to tune related hyperparameters of GraphStorm so as to achieve best performance.


[WIP]Scale to Enterprise-level Graph
--------------------------------

For acquainted users who wish to use their own Graph Neural Network models in GraphStorm,

- Run GraphStorm in Distributed running environment.
- Run GraphStorm in SageMaker Distributed running environment. 

[WIP]Avanced Topics
--------------

- Use Your Own GNN Models
- APIs

Contribution
-------------
GraphStorm is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. We welcome contributions.
Join us on `GitHub <https://github.com/awslabs/graphstorm>`_.

Indices
=======

* :ref:`genindex`
