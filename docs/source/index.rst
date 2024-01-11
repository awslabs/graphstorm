
Welcome to the GraphStorm Documentation and Tutorials
=======================================================

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
   :caption: Distributed Processing
   :hidden:
   :glob:

   gs-processing/gs-processing-getting-started
   gs-processing/usage/example
   gs-processing/usage/distributed-processing-setup
   gs-processing/usage/amazon-sagemaker
   gs-processing/usage/emr-serverless
   gs-processing/developer/input-configuration

.. toctree::
   :maxdepth: 1
   :caption: Distributed Training
   :hidden:
   :glob:

   scale/distributed
   scale/sagemaker

.. toctree::
   :maxdepth: 1
   :caption: Programming User Guide
   :hidden:
   :glob:

   notebooks/Notebook_0_Data_Prepare

.. toctree::
   :maxdepth: 1
   :caption: Advanced Topics
   :hidden:
   :glob:

   advanced/own-models
   advanced/language-models
   advanced/advanced-usages
   advanced/advanced-wholegraph

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:
   :glob:

   api/graphstorm
   api/graphstorm.dataloading
   api/graphstorm.eval
   api/graphstorm.inference
   api/graphstorm.model
   api/graphstorm.trainer

GraphStorm is a graph machine learning (GML) framework designed for enterprise use cases. It simplifies the development, training and deployment of GML models on industry-scale graphs (measured in billons of nodes and edges) by providing scalable training and inference pipelines of GML models. GraphStorm comes with a collection of built-in GML models, allowing users to train a GML model with a single command, eliminating the need to write any code. Moreover, GraphStorm provides a wide range of configurations to customiz model implementations and training pipelines, enhancing model performance. In addition, GraphStorm offers a programming interface that enables users to train custom GML models in a distributed manner. Users can bring their own model implementations and leverage the GraphStorm training pipeline for scalability.

Getting Started
----------------

For beginners, please first start with the :ref:`GraphStorm Docker environment setup<setup>`. This tutorial covers how to set up a Docker environment and build a GraphStorm Docker image, which serves as the Standalone running environment for GraphStorm. We are working on supporting more running environments for GraphStorm.

Once successfully set up the GraphStorm Docker running environment,

- follow the :ref:`GraphStorm Standalone Mode Quick-Start Tutorial<quick-start-standalone>` to run examples using GraphStorm built-in data and models, hence getting familiar with GraphStorm's usage of training and inference.
- follow the :ref:`Use Your Own Graph Data Tutorial<use-own-data>` to prepare your own graph data for using GraphStorm.
- read the :ref:`GraphStorm Training and Inference Configurations<configurations-run>` to learn the various configurations provided by GraphStorm that can help to achieve the best performance.

Scale to Giant Graphs
---------------------------------

For users who wish to train and run infernece on very large graphs,

- follow the :ref:`Use GraphStorm Distributed Data Processing<gs-processing>` tutorial to process and construction large graphs in the Distributed mode.
- follow the :ref:`Use GraphStorm in a Distributed Cluster<distributed-cluster>` tutorial to use GraphStorm in the Distributed mode.
- follow the :ref:`Use GraphStorm on SageMaker<distributed-sagemaker>` tutorial to use GraphStorm in the Distribute mode based on Amazon SageMaker.

Advanced Topics
--------------------

- For users who want to use their own GML models in GraphStorm, follow the :ref:`Use Your Own GNN Models<use-own-models>` tutorial to learn the programming interfaces and the steps of how to modify users' own models.
- For users who want to leverage language models on nodes with text features, follow the :ref:`Use Language Model in GraphStorm<language_models>` tutorial to learn how to leverage BERT models to use text as node features in GraphStorm.
- There are various usages of GraphStorm to both speed up training process and help to boost model performance. Users can find these usages in the :ref:`Advanced Usages<advanced_usages>` page.

Contribution
-------------
GraphStorm is free software; you can redistribute it and/or modify it under the terms of the Apache License 2.0. We welcome contributions.
Join us on `GitHub <https://github.com/awslabs/graphstorm>`_.
