.. _using-graphbolt-ref:

Using GraphBolt to speed up training and inference
==================================================

With GraphStorm ``v0.4``, we introduced support for
`GraphBolt <https://docs.dgl.ai/stochastic_training/>`_
stochastic training. GraphBolt is a new data loading module for DGL that enables faster and more
efficient graph sampling, potentially leading to significant performance benefits.

In this guide we'll give an example of how to prepare your data for use with GraphBolt.

Pre-requisites
--------------

To use GraphBolt you need to have a ``dgl >= 2.1.0`` installed. We recommend using DGL 2.3.0
with PyTorch 2.3 for better compatibility with GraphStorm. To install it you can use:

.. code-block:: bash

    pip install dgl==2.3.0 -f https://data.dgl.ai/wheels/torch-2.3/repo.html

where you can replace ``torch-2.3`` with the version of PyTorch you have installed.
For more details on DGL installation see the `DGL docs <https://www.dgl.ai/pages/start.html>`_.

If using one of the `GraphStorm images <https://github.com/awslabs/graphstorm/blob/main/docker/local/Dockerfile.local>`_
for version ``v0.4``, you should already have the environment set up correctly.

To follow along this tutorial you will also need to have installed GraphStorm and cloned
the GraphStorm repository:

.. code-block:: bash

    pip install graphstorm
    git clone https://github.com/awslabs/graphstorm.git
    cd graphstorm

.. note:: Ensure your environment is set up with enough shared memory.
    Because DGL uses shared memory to store graph structures and features,
    you need to ensure there's enough shared memory configured for your
    system. For most Linux systems it's automatically set up as a ``tempfs`` volume
    using part of main memory (``df -h | grep shm``). If however you're using a Docker container
    to run the example, ensure you're giving the container enough shared memory:
    ``docker run --shm-size=16g -it graphstorm:local-cpu /bin/bash``

Preparing data for use with GraphBolt
-------------------------------------

In order to use GraphBolt for training, we need to first convert our DGL data to the GraphBolt
format.

If you haven't created the partitioned graph data using GConstuct or GSPartition yet, you can
simply provide the additional argument ``--use-graphbolt true`` when creating/partitioning your
graph.

We will be following along the example from :ref:`use-own-data` using the same dataset.

Once we have set up our environment as above, and our current working directory is the
root of the GraphStorm repository we can run:

.. code-block:: bash

    python ./examples/acm_data.py --output-path /tmp/acm_raw

As explained in the tutorial, this will create a set of "raw" data files which we
can use as input to GConstuct to create a partitioned graph.

Now let's generate a partitioned graph (with a single partition), along with GraphBolt
data:

.. code-block:: bash

    python -m graphstorm.gconstruct.construct_graph \
              --conf-file /tmp/acm_raw/config.json \
              --num-parts 1 \
              --graph-name acm \
              --output-dir /tmp/acm_graphbolt \
              --use-graphbolt true

The top level data will look exactly the same as when we run the above without setting ``--use-graphbolt true``:

.. code-block:: bash

    ls /tmp/acm_graphbolt/
    acm.json  data_transform_new.json  edge_label_stats.json  edge_mapping.pt  node_label_stats.json  node_mapping.pt  part0  raw_id_mappings

GraphBolt will create a modified representation of each graph partition:

.. code-block:: bash

    ls /tmp/acm_graphbolt/part0
    edge_feat.dgl  fused_csc_sampling_graph.pt  graph.dgl  node_feat.dgl

The file ``fused_csc_sampling_graph.pt`` is the new representation that GraphBolt will use during training
to optimized data loading.

GSPartition with GraphBolt
^^^^^^^^^^^^^^^^^^^^^^^^^^


Similar to GConstuct, when running distributed graph partitioning with the GSPartition module you can
provide the ``--use-graphbolt true`` argument to convert the resulting partitions to GraphBolt
at the end of partitioning. You'll need your data to be in DGL's "chunked graph format",
which can be produced by feeding your raw data and GConstuct configuration to
:ref:`GSProcessing <gsprocessing_prerequisites_index>`.

.. code-block:: bash

    python -m graphstorm.gpartition.dist_partition_graph \
        --input-path "${INPUT_PATH}" \
        --ip-config ip_list.txt \
        --metadata-filename chunked_graph_meta.json \
        --num-parts 2 \
        --output-path "$DIST_GRAPHBOLT_PATH" \
        --ssh-port 2222 \
        --use-graphbolt "true"


Running training and inference tasks with GraphBolt enabled
-----------------------------------------------------------

Now that we have prepared our data we can run a training job, with GraphBolt enabled.
We run the same node classification task as in the original guide:

.. code-block:: bash

    python -m graphstorm.run.gs_node_classification \
              --part-config /tmp/acm_graphbolt/acm.json \
              --num-epochs 30 \
              --num-trainers 1 \
              --num-servers 1 \
              --cf ./examples/use_your_own_data/acm_nc.yaml \
              --save-model-path /tmp/acm_nc_graphbolt/models \
              --node-feat-name paper:feat author:feat subject:feat \
              --use-graphbolt true

Similarly, we can run an inference task with GraphBolt enabled as such:

.. code-block:: bash

    python -m graphstorm.run.gs_node_classification \
              --inference \
              --part-config /tmp/acm_graphbolt/acm.json \
              --num-trainers 1 \
              --num-servers 1 \
              --cf ./examples/use_your_own_data/acm_nc.yaml \
              --node-feat-name paper:feat author:feat subject:feat \
              --restore-model-path /tmp/acm_nc_baseline/models/epoch-29 \
              --save-prediction-path  /tmp/acm_nc_graphbolt/predictions \
              --use-graphbolt true

Converting existing partitioned graphs to the GraphBolt format
--------------------------------------------------------------

If you have an existing partitioned graph that you would like to
run training or inference on, using GraphBolt, you can directly convert that
data using our GraphBolt conversion entry point:

.. code-block:: bash

    # For testing, let's remove the converted graph file
    rm /tmp/acm_graphbolt/part0/fused_csc_sampling_graph.pt
    # Now let's run the standalone GraphBolt conversion
    python -m graphstorm.gpartition.convert_to_graphbolt \
        --input-path /tmp/acm_graphbolt
        --metadata-filename acm.json
    # We'll see the GraphBolt representation has been re-created
    ls /tmp/acm_graphbolt/part0
    edge_feat.dgl  fused_csc_sampling_graph.pt  graph.dgl  node_feat.dgl
