.. _gsprocessing_prerequisites_index:

========================================
GraphStorm Distributed Data Processing
========================================

GraphStorm Distributed Data Processing (GSProcessing) enables the processing and preparation of massive graph data for training with GraphStorm. GSProcessing handles generating unique node IDs, encoding edge structure files, processing individual features, and preparing data for the distributed partition stage.

.. note::

    * We use PySpark for horizontal parallelism, enabling scalability to graphs with billions of nodes and edges.
    * GraphStorm currently only supports running GSProcessing on AWS Infras including `Amazon SageMaker <https://docs.aws.amazon.com/sagemaker/>`_, `EMR Serverless <https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/emr-serverless.html>`_, and `EMR on EC2 <https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-what-is-emr.html>`_.

The following sections outline essential prerequisites and provide a detailed guide to use 
GSProcessing.
The first section provides an introduction to GSProcessing, how to install it locally and a quick example of its input configuration.
The second section demonstrates how to set up GSProcessing for distributed processing, enabling scalable and efficient processing using AWS resources.
The third section explains how to deploy GSProcessing job with AWS infrastructure. The last section offers the details about generating a configuration file for GSProcessing jobs.

.. toctree::
  :maxdepth: 1
  :titlesonly:

  gs-processing-getting-started.rst
  distributed-processing-setup.rst
  aws-infra/index.rst
  input-configuration.rst
