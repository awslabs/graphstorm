.. _using-graphbolt-ref:

Using GraphBolt to speed up training and inference
==================================================

With GraphStorm ``v0.4``, we introduced support for
`GraphBolt <https://docs.dgl.ai/stochastic_training/>`_
stochastic training. GraphBolt is a new data loading module for DGL that enables faster and more
efficient graph sampling, potentially leading to significant efficiency benefits.

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
    to run the example, ensure you're giving the container enough shared memory
    by mapping it to the host's shared memory, like:
    ``docker run -v /dev/shm:/dev/shm -it graphstorm:local-cpu /bin/bash``

Preparing data for use with GraphBolt
-------------------------------------

In order to use GraphBolt for training, we need to first convert our graph data to the GraphBolt
format.
If you are going to create the partitioned graph data using GConstuct or GSPartition, you can
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
to optimize data loading.

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
        --use-graphbolt true


Running training and inference tasks with GraphBolt enabled
-----------------------------------------------------------

Now that we have prepared our data we can run a training job, with GraphBolt enabled.
We run the :ref:`same node classification task as in the original guide <launch_training_oyog>`
by adding the `--use-graphbolt true` argument to enable GraphBolt:

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

If you have a partitioned graph that you would like to
run training or inference on, using GraphBolt, you can directly convert that
data using our GraphBolt conversion entry point:

.. code-block:: bash

    # For testing, let's remove the converted graph file
    rm /tmp/acm_graphbolt/part0/fused_csc_sampling_graph.pt
    # Now let's run the standalone GraphBolt conversion
    python -m graphstorm.gpartition.convert_to_graphbolt \
        --metadata-filepath /tmp/acm_graphbolt/acm.json
    # We'll see the GraphBolt representation has been re-created
    ls /tmp/acm_graphbolt/part0
    edge_feat.dgl  fused_csc_sampling_graph.pt  graph.dgl  node_feat.dgl

Using GraphBolt on SageMaker
----------------------------

Before being able to train on SageMaker
we need to ensure our data on S3 have been
converted to the GraphBolt format.
When using GConstruct to process our data
we can include the GraphBolt data conversion in the GConstruct
step as we'll show below.

When using distributed graph construction with GSProcessing and GSPartition,
to prepare data to use with GraphBolt on SageMaker
we need to launch the GraphBolt data conversion step
as a separate SageMaker job, after
the partitioned DGL graph files have been created on S3.

After running your distributed partition SageMaker job as normal using
``sagemaker/launch_partition.py``, you next need to launch the
``sagemaker/launch_graphbolt_convert.py`` script, passing as input
the S3 URI, where the DistDGL partition data is stored by ``launch_partition.py``,
**plus the suffix `dist_graph`** as that's where GSPartition creates the partition files.

For example, if you used ``--output-data-s3 s3://my-bucket/my-part-graph`` for
``sagemaker/launch_partition.py`` you need to use ``--graph-data-s3 s3://my-bucket/my-part-graph/dist_graph``
for ``sagemaker/launch_graphbolt_convert.py``.

Without using GraphBolt a SageMaker job sequence for distributed processing and training
is ``GSProcessing -> GSPartition -> GSTraining``. To use GraphBolt we need to add
a step after partitioning and before training:
``GSProcessing -> GSPartition -> GraphBoltConvert -> GSTraining``.

.. code-block:: bash

    cd graphstorm/sagemaker
    sagemaker/launch_partition.py \
        --graph-data-s3 "s3-uri-where-gsprocessing-data-exist" \
        --output-data-s3 "s3-uri-where-gspartition-data-will-be"
        # Add other required parameters like --partition-algorithm, --num-instances etc.

    # Once the above job succeeds we run the following command to convert the data to GraphBolt format.
    # Note the /dist_graph suffix!
    sagemaker/launch_graphbolt_convert.py \
        --graph-data-s3 "s3-uri-where-gspartition-data-will-be/dist_graph" \
        --metadata-filename "metadata.json" # Or <graph-name>.json for gconstruct-ed partitions


If your data are small enough to process on a single SageMaker instance
using ``GConstuct``, you can simply pass the ``--use-graphbolt true`` argument
to the ``GConstruct`` SageMaker launch script and that will create the
necessary GraphBolt files as well.
So the job sequence there remains ``GConstruct -> GSTraining``.

.. code-block:: bash

    sagemaker/launch_gconstruct.py \
        --graph-data-s3 "s3-uri-where-raw-data-exist" \
        --output-data-s3 "s3-uri-where-gspartition-data-will-be" \
        --graph-config-file "gconstruct-config.json" \
        --use-graphbolt true

If you initially used GConstruct to create the non-GraphBolt DistDGL files,
you'll need to pass in the additional argument ``--metadata-filename``
to ``launch_graphbolt_convert.py``.
Use ``<graph-name>.json`` where the graph name should be the
one you used with GConstruct as shown below:

.. code-block:: bash

    # NOTE: we provide 'my-graph' as the graph name
    sagemaker/launch_gconstruct.py \
        --graph-name my-graph \
        --graph-data-s3 "s3-uri-where-raw-data-exist" \
        --output-data-s3 "s3-uri-where-gspartition-data-will-be" \
        --graph-config-file "gconstruct-config.json" # We don't add --use-graphbolt true

    # Once the above job succeeds we run the below to convert the data to GraphBolt
    # NOTE: Our metadata file name will be named 'my-graph.json'
    sagemaker/launch_graphbolt_convert.py \
        --graph-data-s3 "s3-uri-where-gspartition-data-will-be"
        --metadata-filename "my-graph.json" # Should be <graph-name>.json


Once the data have been converted to the GraphBolt format you can run your training
and inference jobs as before, passing the additional
argument ``--use-graphbolt`` to the SageMaker launch scripts
to indicate that we want to use GraphBolt during training/inference:

.. code-block:: bash

    sagemaker/launch_train.py \
        --graph-name my-graph \
        --graph-data-s3 "s3-uri-where-gspartition-data-will-be" \
        --yaml-s3 "s3-path-to-train-yaml" \
        --use-graphbolt true


If you want to test steps locally you can use SageMaker's
[local mode](https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode)
by providing `local` as the instance type in the launch scripts.
