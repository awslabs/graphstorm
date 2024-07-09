==============================
Distributed Graph Construction
==============================

GraphStorm Distributed Data Processing (GSProcessing) allows you to process
and prepare massive graph data for training with GraphStorm. GSProcessing takes
care of generating unique ids for nodes, using them to encode edge structure files,
process individual features and prepare the data to be passed into the distributed
partitioning and training pipeline of GraphStorm.

We use PySpark to achieve horizontal parallelism, allowing us to scale to graphs with billions of nodes and edges.

The following sections provide comprehensive guidance on effectively utilizing GSProcessing.
The first section details the installation process for GSProcessing.
The second section offers examples on drafting a configuration file for a GSProcessing job.
The final section explains how to deploy your GSProcessing job into AWS infrastructure.

.. toctree::
   :maxdepth: 2
   :glob:

   gs-processing-prerequisites-index.rst
   ../developer/input-configuration
   aws-infra-index.rst
   ../usage/example.rst