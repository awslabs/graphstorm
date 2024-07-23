.. _gpartition_index:

===================================
Running partition jobs on AWS Infra
===================================

GraphStorm Distributed Data Partition (GPartition) allows you to do distributed partition on preprocessed graph data
prepared by :ref:`GSProcessing<gsprocessing_prerequisites_index>`. To enable distributed training, the preprocessed input data must be converted to a partitioned graph representation.
GPartition allows user to handle massive graph data in distributed clusters on AWS. GPartition is built on top of the
dgl `distributed graph partitioning pipeline <https://docs.dgl.ai/en/latest/guide/distributed-preprocessing.html#distributed-graph-partitioning-pipeline>`_.

GPartition consists of two steps: Graph Partitioning and Data Dispatching. Graph Partitioning step will calculate the ownership of
each partition and save the results as a set of files called partition assignment. Data Dispatching step will physically partitions the
graph data and dispatch them to the machines user specified. It will generate the graph data in DGL format for distributed training.

The first section includes instructions on how to run GPartition on `Amazon SageMaker <https://docs.aws.amazon.com/sagemaker/>`_.
The second section includes instructions on how to run GPartition on `Amazon EC2 clusters <https://docs.aws.amazon.com/AmazonECS/latest/developerguide/clusters.html>`_.

.. toctree::
   :maxdepth: 1
   :glob:

   sagemaker.rst
   ec2-clusters.rst
