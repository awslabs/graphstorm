.. _model_training_inference:

========================================
GraphStorm Model Training and Inference
========================================

Once your raw data are converted into partitioned DGL distributed graphs by using the :ref:`GraphStorm Graph Construction <graph_construction>` user guide, you can use Graphstorm CLIs to train GML models and do inference on a signle machine if there is one partition only, or on a distributed environment, such as a Linux cluster, for multiple partition graphs.

This section provides guidelines of GraphStorm model training and inference on :ref:`signle machine <single-machine-training-inference>`, :ref:`distributed clusters <distributed-cluster>`, and :ref:`Amazon SageMaker <distributed-sagemaker>`.

GraphStorm CLIs require less- or no-code operations for users to perform Graph Machine Learning (GML) tasks. In most cases, users only need to configure the parameters or arguments provided by GraphStorm to fulfill their GML tasks. Users can find the details of these configurations in the :ref:`Model Training and Inference Configurations<configurations-run>`.

In scenarios when inference time is critical, such as detecting fraudulent transactions in seconds, Graphstorm can deploy 
a trained model with Amazon SageMaker endpoint for real time inference. :ref:`Graphstorm real-time inference 
user guide <real-time-inference-on-sagemaker>` explains the deployment steps. 
And the :ref:`specification of real-time inference request and response <real-time-inference-spec>` provides 
details of contents of endpoint invoking requests and responses.

In addition, there are two node ID mapping operations during the graph construction procedure, and these mapping results are saved in a certain folder by which GraphStorm training and inference CLIs will automatically use to remap prediction results' node IDs back to the original IDs. In case when such automatic remapping does not occur, you can find the details of outputs of model training and inference without remapping in :ref:`GraphStorm Training and Inference Output <gs-output>`. In addition, users can do the remapping mannually according to the :ref:`GraphStorm Output Node ID Remapping <gs-output-remapping>` guideline.

.. toctree::
   :maxdepth: 2
   :glob:

   single-machine-training-inference
   distributed/cluster
   distributed/sagemaker
   configuration-run
   real-time-inference
   real-time-inference-spec
   output
   output-remapping
