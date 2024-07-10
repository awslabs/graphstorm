..  _row_count_alignment:

Row count alignment
===================

After the data processing step we need to perform an additional step
to ensure that our processed data conform to the assumptions of the distributed
partitioning pipeline. In particular DistPartitioning expects that:

* For each node/edge type:
    * Every file output has the same number of files.
        * For example, for an edge type ``x:to:y``, that had
          two features, ``feat1`` and ``feat2``, the number
          (partition count) of the files produced separately
          for ``feat1``, ``feat2`` and the edge structure
          needs to be the same.
    * Each respective file in the output has the same row count.
        * For example, assuming ``feat1``, ``feat2``, and ``edges``
          had 2 part-files each, the number of rows in file-part-1
          needs to be the same across all three file sources, and the
          number of rows in file-part-2 needs to be the same
          across all three file sources.


In code the above means:

.. code-block:: python

    files_for_feat1 = os.listdir("edge_data/x:to:y-feat1/")
    files_for_feat2 = os.listdir("edge_data/x:to:y-feat2/")
    files_for_edges = os.listdir("edges/x:to:y")

    num_feat1_files = len(files_for_feat1)
    num_feat2_files = len(files_for_feat2)
    num_edges_files = len(files_for_edges)

    assert num_feat1_files == num_feat2_files == num_edges_files

In addition, for each node/edge type, the row counts of each respective file
in their output needs to match, i.e.:

.. code-block:: python

    from pyarrow import parquet as pq

    row_counts_feat1 = [pq.read_metadata(fpath).num_rows for fpath in files_for_feat1]
    row_counts_feat2 = [pq.read_metadata(fpath).num_rows for fpath in files_for_feat2]
    row_counts_edges = [pq.read_metadata(fpath).num_rows for fpath in files_for_edges]

    assert row_counts_feat1 == row_counts_feat2 == row_counts_edges

Note that these assumptions only apply `per type`; file counts and per-file
row counts do not need to match between different node/edge types.

Because of the distributed and speculative nature of Spark execution, it's
not possible to guarantee that the row counts will match between the file
outputs we create for every node types features, or the structure and
features of an edge type.

Therefore and additional step which we call `repartitioning` is necessary
after the processing step. This step performs two functions:

1. Align the row counts for each edge/node type.
2. Ensure that data shapes for masks and labels match what
   what DistPartitioning expects, which are flat ``(N,)`` arrays,
   instead of what Spark produces which is ``(N, 1)`` Parquet output.

Local repartitioning
--------------------

The simplest way to apply the re-partitioning step is to do so during the `gs-processing` step,
by passing the additional `--do-repartition True` argument to our launch script.

Alternatively, we can run a local re-partitioning job using a local
installation of GSProcessing:

.. code-block:: bash

    gs-repartition --input-prefix local_or_s3_path_to_processed_data

The repartitioning command will call the ``graphstorm_processing/repartition_files.py``
Python script and execute the step locally. The script only requires the
``input-prefix`` argument to function, but provides optional arguments
to customize the input/output file names and whether to use an
in-memory or file streaming implementation for row-count alignment.

You can use `gs-repartition --help` for more details on the arguments.

Repartitioning on SageMaker
---------------------------

To avoid local processing it is also possible to run re-partitioning on
SageMaker. You would need to complete the steps described in
:doc:`distributed-processing-setup` to build and push a SageMaker
ECR image, and then you're able to launch the re-partitioning job
on SageMaker:

.. code-block:: bash

    bash docker/build_gsprocessing_image.sh --environment sagemaker --region ${REGION}
    bash docker/push_gsprocessing_image.sh --environment sagemaker --region ${REGION}

    SAGEMAKER_ROLE_NAME="enter-your-sagemaker-execution-role-name-here"
    IMAGE_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/graphstorm-processing-sagemaker:latest-x86_64"
    ROLE="arn:aws:iam::${ACCOUNT}:role/service-role/${SAGEMAKER_ROLE_NAME}"
    INSTANCE_TYPE="ml.t3.xlarge"

    python scripts/run_repartitioning.py --s3-input-prefix ${PROCESSED_OUTPUT} \
        --role ${ROLE} --image ${IMAGE_URI} \
        --instance-type ${INSTANCE_TYPE} --wait-for-job

File streaming repartitioning
-----------------------------

The default implementation of re-partitioning will load each
feature/edge type in memory and perform the row-count alignment.
Using SageMaker Processing with instances such as ``ml.r5.24xlarge``
with 768GB of memory, you should be able to process data with
billions of nodes/edges and hundreds of features.

If however your data are so large that they cause out-of-memory
errors even on SageMaker, you can use the file streaming implementation
of re-partitioning, which should allow you to scale to any file size.

To do so, simply modify your call to include:

.. code-block:: bash

    gs-repartition --input-prefix local_or_s3_path_to_processed_data \
        --streaming-repartitioning True

A similar modification can be applied to the SageMaker launch call:

.. code-block:: bash

    python scripts/run_repartitioning.py --s3-input-prefix ${PROCESSED_OUTPUT} \
        --role ${ROLE} --image ${IMAGE_URI}  --config-filename "metadata.json" \
        --instance-type ${INSTANCE_TYPE} --wait-for-job \
        --streaming-repartitioning True

The file streaming implementation will hold at most 2 files worth of data
in memory, so by choosing an appropriate file number when processing you should
be able to process any data size.

.. note::

    The file streaming implementation will be much slower than the in-memory
    one, so only use in case no instance size can handle your data.
