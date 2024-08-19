.. _gsprocessing_sagemaker:

Running distributed jobs on Amazon SageMaker
============================================

Once the :ref:`Amazon SageMaker Setup<gsprocessing_distributed_setup>` is complete, we can
use the Amazon SageMaker launch scripts to launch distributed processing
jobs that use AWS resources.

To demonstrate the usage of GSProcessing on Amazon SageMaker, we will execute the same job we used in our local
execution example, but this time use Amazon SageMaker to provide the compute resources instead of our
local machine.

Before starting make sure you have uploaded the input data as described in :ref:`gsp-upload-data-ref`.

Launch the GSProcessing job on Amazon SageMaker
-----------------------------------------------

Once the data are uploaded to S3, we can use the Python script
``graphstorm-processing/scripts/run_distributed_processing.py``
to run a GSProcessing job on Amazon SageMaker.

For this example we'll use a SageMaker Spark cluster with 2 ``ml.t3.xlarge`` instances
since this is a tiny dataset. Using SageMaker you'll be able to create clusters
of up to 20 instances, allowing you to scale your processing to massive graphs,
using larger instances like ``ml.r5.24xlarge``.

Since we're now executing on AWS, we'll need access to an execution role
for SageMaker and the ECR image URI we created in :ref:`GSProcessing distributed setup<gsprocessing_distributed_setup>`.
For instructions on how to create an execution role for SageMaker
see the `AWS SageMaker documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html#sagemaker-roles-create-execution-role>`_.

Let's set up a small ``bash`` script that will run the parametrized processing
job, followed by the re-partitioning job, both on SageMaker:

.. code-block:: bash

    ACCOUNT="enter-your-account-id-here" # e.g 1234567890
    MY_BUCKET="enter-your-bucket-name-here"
    SAGEMAKER_ROLE_NAME="enter-your-sagemaker-execution-role-name-here"
    REGION="bucket-region" # e.g. us-west-2
    INPUT_PREFIX="s3://${MY_BUCKET}/gsprocessing-input"
    OUTPUT_BUCKET=${MY_BUCKET}
    GRAPH_NAME="small-graph"
    CONFIG_FILE="gconstruct-config.json"
    INSTANCE_COUNT="2"
    INSTANCE_TYPE="ml.t3.xlarge"
    NUM_FILES="-1"

    IMAGE_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/graphstorm-processing-sagemaker:latest-x86_64"
    ROLE="arn:aws:iam::${ACCOUNT}:role/service-role/${SAGEMAKER_ROLE_NAME}"

    OUTPUT_PREFIX="s3://${OUTPUT_BUCKET}/gsprocessing/sagemaker/${GRAPH_NAME}/${INSTANCE_COUNT}x-${INSTANCE_TYPE}-${NUM_FILES}files/"

    # This will run and block until the GSProcessing job is done
    python scripts/run_distributed_processing.py \
        --s3-input-prefix ${INPUT_PREFIX} \
        --s3-output-prefix ${OUTPUT_PREFIX} \
        --role ${ROLE} \
        --image ${IMAGE_URI} \
        --region ${REGION} \
        --config-filename ${CONFIG_FILE} \
        --instance-count ${INSTANCE_COUNT} \
        --instance-type ${INSTANCE_TYPE} \
        --job-name "${GRAPH_NAME}-${INSTANCE_COUNT}x-${INSTANCE_TYPE//./-}-${NUM_FILES}files" \
        --num-output-files ${NUM_FILES} \
        --do-repartition True \
        --wait-for-job

Launch the gs-repartition job on Amazon SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


In the above we have set `--do-repartition True` to perform the re-partition step on the Spark
leader instance, since we know each individual feature and the edge structure are small
enough to fit in the memory of the Spark leader.
For large graphs you will
want to launch that step as a separate job on an instance with more memory to avoid memory errors.
`ml.r5` instances should allow you to re-partition graph data with billions of nodes and edges.
For more details on the re-partitioning step see :ref:`row count alignment<row_count_alignment>`.

To run the re-partition job as a separate job use:

.. code-block:: bash

    # Ensure the bash variables are as set as above.
    # This will only run the follow-up re-partitioning job on a single instance
    python scripts/run_repartitioning.py --s3-input-prefix ${OUTPUT_PREFIX} \
        --role ${ROLE} --image ${IMAGE_URI}  --config-filename "metadata.json" \
        --instance-type ${INSTANCE_TYPE} --wait-for-job


The ``--num-output-files`` parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can see that we provided a parameter named
``--num-output-files`` to ``run_distributed_processing.py``. This is an
important parameter, as it provides a hint to set the parallelism for Spark.

We recommend setting this to `-1` to let Spark decide the proper value based on the cluster's
vCPU count. If setting it yourself a good value to use is
``num_instances * num_cores_per_instance * 2``, which will ensure good
utilization of the cluster resources. For EMR serverless, equivalently set
to ``num_executors * num_cores_per_executor * 2``


Examine the output
------------------

Once both jobs are finished we can examine the output created, which
should match the output we saw when running the same jobs locally
in :ref:`gsp-examining-output`.


.. code-block:: bash

    $ aws s3 ls ${OUTPUT_PREFIX}

                               PRE edges/
                               PRE node_data/
                               PRE raw_id_mappings/
    2023-08-05 00:47:36        804 launch_arguments.json
    2023-08-05 00:47:36      11914 metadata.json
    2023-08-05 00:47:37        545 perf_counters.json
    2023-08-05 00:47:37      12082 updated_row_counts_metadata.json

Run distributed partitioning and training on Amazon SageMaker
-------------------------------------------------------------

With the data now processed you can follow the
`GraphStorm Amazon SageMaker guide
<https://graphstorm.readthedocs.io/en/latest/scale/sagemaker.html#run-graphstorm-on-sagemaker>`_
to partition your data and run training on AWS.
