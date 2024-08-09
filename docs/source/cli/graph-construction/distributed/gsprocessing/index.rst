.. _gsprocessing_prerequisites_index:

===============================================
Distributed Graph Processing
===============================================

GraphStorm Distributed Data Processing (GSProcessing) allows you to process
and prepare massive graph data for training with GraphStorm. GSProcessing takes
care of generating unique ids for nodes, using them to encode edge structure files,
process individual features and prepare the data to be passed into the distributed
partitioning and training pipeline of GraphStorm.

We use PySpark to achieve horizontal parallelism, allowing us to scale to graphs with billions of nodes and edges.

.. note::
    GraphStorm currently only supports running GSProcessing on AWS Infras including `Amazon SageMaker <https://docs.aws.amazon.com/sagemaker/>`_, `EMR Serverless <https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/emr-serverless.html>`_, and `EMR on EC2 <https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-what-is-emr.html>`_.

The following sections outline essential prerequisites and provide a detailed guide to use 
GSProcessing.
The first section provides an introduction to GSProcessing, how to install it locally and a quick example of its input configuration.
The second section demonstrates how to set up GSProcessing for distributed processing, enabling scalable and efficient processing using AWS resources.

.. toctree::
  :maxdepth: 1
  :titlesonly:

  gs-processing-getting-started.rst
  distributed-processing-setup.rst
  aws-infra/index.rst
  input-configuration.rst
