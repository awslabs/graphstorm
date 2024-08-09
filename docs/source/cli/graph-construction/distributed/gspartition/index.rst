.. _gspartition_index:

=======================================
GraphStorm Distributed Graph Partition
=======================================

GraphStorm Distributed Graph Partition (GSPartition), which is built on top of the
dgl `distributed graph partitioning pipeline <https://docs.dgl.ai/en/latest/guide/distributed-preprocessing.html#distributed-graph-partitioning-pipeline>`_, allows users to do distributed partition on the outputs of :ref:`GSProcessing<gs-processing>`. 

GSPartition consists of two steps: Graph Partitioning and Data Dispatching. Graph Partitioning step assigns each node to one partition and save the results as a set of files, called partition assignment. Data Dispatching step will physically partition the
graph data and dispatch them according to the partition assignment. It will generate the graph data in DGL distributed graph format, ready for GraphStorm distributed training and inference.

.. note::
    GraphStorm currently only supports running GSPartition on AWS infrastructure, i.e., `Amazon SageMaker <https://docs.aws.amazon.com/sagemaker/>`_ and `Amazon EC2 clusters <https://docs.aws.amazon.com/AmazonECS/latest/developerguide/clusters.html>`_. But, users can easily create your own Linux clusters by following the GSPartition tutorial on Amazon EC2.

The first section includes instructions on how to run GSPartition on `Amazon SageMaker <https://docs.aws.amazon.com/sagemaker/>`_.
The second section includes instructions on how to run GSPartition on `Amazon EC2 clusters <https://docs.aws.amazon.com/AmazonECS/latest/developerguide/clusters.html>`_.

.. toctree::
   :maxdepth: 1
   :glob:

   sagemaker.rst
   ec2-clusters.rst
