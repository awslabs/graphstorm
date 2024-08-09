.. _model_training_inference:

========================================
GraphStorm Model Training and Inference
========================================

Once users convert your raw data into the partitioned DGL distributed graphs by using the :ref:`GraphStorm Graph Construction <graph_construction>` user guide, you can use Graphstorm CLIs to train GML models and do inference on a signle machine if there is one partition only, or on a distributed environment, such as a Linux cluster, for multiple partition graphs.

This section provides guidelines of GraphStorm model training and inference on :ref:`signle machine <single-machine-training-inference>`, :ref:`distributed clusters <distributed-cluster>`, and :ref:`Amazon SageMaker <distributed-sagemaker>`.

In addition, there are two node ID mapping operations during the graph construction procedure, and these mapping results are saved in a certain folder by which GraphStorm inference pipelines will automatically use to remap prediction results' node IDs back to the original IDs. In case when such automatic remapping does not occure, users can do it mannually according to the :ref:`GraphStorm Output Node ID Remapping <output-remapping>` guideline.

.. toctree::
   :maxdepth: 2
   :glob:

   single-machine-training-inference
   distributed/cluster
   distributed/sagemaker
   output-remapping
