===================================
Running partition jobs on AWS Infra
===================================

GraphStorm Distributed Data Partition (GPartition) allows you to do distributed partition on preprocessed graph data
prepared by :ref:`GSProcessing<gsprocessing_prerequisites_index>`. Before entering the GraphStorm training pipeline,
the preprocessed input data must be partitioned.
GPartition offers to handle massive graph data in distributed clusters on AWS Infras. GPartition is built beyond the
dgl `distributed graph partitioning pipeline <https://docs.dgl.ai/en/latest/guide/distributed-preprocessing.html#distributed-graph-partitioning-pipeline>`_.

The first section includes instructions on how to run GPartition on `Amazon SageMaker <https://docs.aws.amazon.com/sagemaker/>`_.
The second section includes instructions on how to run GPartition on `Amazon EC2 clusters <https://docs.aws.amazon.com/AmazonECS/latest/developerguide/clusters.html>`_.

.. toctree::
   :maxdepth: 1
   :glob:

   sagemaker.rst
   ec2-clusters.rst
