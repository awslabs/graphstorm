==============================
Distributed Graph Construction
==============================

GraphStorm Distributed Data Processing (GSProcessing) allows you to process
and prepare massive graph data for training with GraphStorm. GSProcessing takes
care of generating unique ids for nodes, using them to encode edge structure files,
process individual features and prepare the data to be passed into the distributed
partitioning and training pipeline of GraphStorm.

We use PySpark to achieve horizontal parallelism, allowing us to scale to graphs with billions of nodes and edges.

The following sections provide guidance on effectively utilizing GSProcessing.
The first section details the execution environment setup for GSProcessing.
The second section offers examples on drafting a configuration file for a GSProcessing job.
The third section explains how to deploy your GSProcessing job with AWS infrastructure.
The final section shows an example to quick start GSProcessing.

.. toctree::
   :maxdepth: 2
   :glob:

   gs-processing-prerequisites-index.rst
   ../developer/input-configuration
   aws-infra-index.rst
   ../usage/example.rst