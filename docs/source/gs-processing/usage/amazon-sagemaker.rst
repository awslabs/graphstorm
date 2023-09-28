Running distributed jobs on Amazon SageMaker
============================================

Once the :doc:`distributed processing setup <distributed-processing-setup>` is complete, we can
use the Amazon SageMaker launch scripts to launch distributed processing
jobs that use AWS resources.

To demonstrate the usage of GSProcessing on Amazon SageMaker, we will execute the same job we used in our local
execution example, but this time use Amazon SageMaker to provide the compute resources instead of our
local machine.

Upload data to S3
-----------------

Amazon SageMaker uses S3 as its storage target, so before starting
we'll need to upload our test data to S3. To do so you will need
to have read/write access to an S3 bucket, and the requisite AWS credentials
and permissions.

We will use the AWS CLI to upload data so make sure it is
`installed <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_
and `configured <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html>`_
in you local environment.

Assuming ``graphstorm/graphstorm-processing`` is our current working
directory we can upload the test data to S3 using:

.. code-block:: bash

    MY_BUCKET="enter-your-bucket-name-here"
    REGION="bucket-region" # e.g. us-west-2
    aws --region ${REGION} s3 sync ./tests/resources/small_heterogeneous_graph/ \
        "${MY_BUCKET}/gsprocessing-input"

.. note::

    Make sure you are uploading your data to a bucket
    that was created in the same region as the ECR image
    you pushed in :doc:`distributed-processing-setup`.


Launch the GSProcessing job on Amazon SageMaker
-----------------------------------------------

Once the data are uploaded to S3, we can use the Python script
``graphstorm-processing/scripts/run_distributed_processing.py``
to run a GSProcessing job on Amazon SageMaker.

For this example we'll use a SageMaker Spark cluster with 2 ``ml.t3.xlarge`` instances
since this is a tiny dataset. Using SageMaker you'll be able to create clusters
of up to 20 instances, allowing you to scale your processing to massive graphs,
using larger instances like `ml.r5.24xlarge`.

Since we're now executing on AWS, we'll need access to an execution role
for SageMaker and the ECR image URI we created in :doc:`distributed-processing-setup`.
For instructions on how to create an execution role for SageMaker
see the `AWS SageMaker documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html#sagemaker-roles-create-execution-role>`_.

Let's set up a small ``bash`` script that will run the parametrized processing
job, followed by the re-partitioning job, both on SageMaker:

.. code-block:: bash

    ACCOUNT="enter-your-account-id-here" # e.g 1234567890
    MY_BUCKET="enter-your-bucket-name-here"
    SAGEMAKER_ROLE_NAME="enter-your-sagemaker-execution-role-name-here"
    REGION="bucket-region" # e.g. us-west-2
    DATASET_S3_PATH="s3://${MY_BUCKET}/gsprocessing-input"
    OUTPUT_BUCKET=${MY_BUCKET}
    DATASET_NAME="small-graph"
    CONFIG_FILE="gconstruct-config.json"
    INSTANCE_COUNT="2"
    INSTANCE_TYPE="ml.t3.xlarge"
    NUM_FILES="4"

    IMAGE_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/graphstorm-processing:0.1.0"
    ROLE="arn:aws:iam::${ACCOUNT}:role/service-role/${SAGEMAKER_ROLE_NAME}"

    OUTPUT_PREFIX="s3://${OUTPUT_BUCKET}/gsprocessing/${DATASET_NAME}/${INSTANCE_COUNT}x-${INSTANCE_TYPE}-${NUM_FILES}files/"

    # Conditionally delete data at output
    echo "Delete all data under output path? ${OUTPUT_PREFIX}"
    select yn in "Yes" "No"; do
        case $yn in
            Yes ) aws s3 rm --recursive ${OUTPUT_PREFIX} --quiet; break;;
            No ) break;;
        esac
    done

    # This will run and block until the GSProcessing job is done
    python scripts/run_distributed_processing.py \
        --s3-input-prefix ${DATASET_S3_PATH} \
        --s3-output-prefix ${OUTPUT_PREFIX} \
        --role ${ROLE} \
        --image ${IMAGE_URI} \
        --region ${REGION} \
        --config-filename ${CONFIG_FILE} \
        --instance-count ${INSTANCE_COUNT} \
        --instance-type ${INSTANCE_TYPE} \
        --job-name "${DATASET_NAME}-${INSTANCE_COUNT}x-${INSTANCE_TYPE//./-}-${NUM_FILES}files" \
        --num-output-files ${NUM_FILES} \
        --wait-for-job

    # This will run the follow-up re-partitioning job
    python scripts/run_repartitioning.py --s3-input-prefix ${OUTPUT_PREFIX} \
        --role ${ROLE} --image ${IMAGE_URI}  --config-filename "metadata.json" \
        --instance-type ${INSTANCE_TYPE} --wait-for-job


.. note::

    The re-partitioning job runs on a single instance, so for large graphs you will
    want to scale up to an instance with more memory to avoid memory errors. `ml.r5` instances
    should allow you to re-partition graph data with billions of nodes and edges.

The ``--num-output-files`` parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can see that we provided a parameter named
``--num-output-files`` to ``run_distributed_processing.py``. This is an
important parameter, as it provides a hint to set the parallelism for Spark.

It can safely be skipped and let Spark decide the proper value based on the cluster's
instance type and count. If setting it yourself a good value to use is
``num_instances * num_cores_per_instance * 2``, which will ensure good
utilization of the cluster resources.


Examine the output
------------------

Once both jobs are finished we can examine the output created, which
should match the output we saw when running the same jobs locally
in :ref:`gsp-examining-output`.


.. code-block:: bash

    $ aws s3 ls ${OUTPUT_PREFIX}

                               PRE edges/
                               PRE node_data/
                               PRE node_id_mappings/
    2023-08-05 00:47:36        804 launch_arguments.json
    2023-08-05 00:47:36      11914 metadata.json
    2023-08-05 00:47:37        545 perf_counters.json
    2023-08-05 00:47:37      12082 updated_row_counts_metadata.json

Run distributed partitioning and training on Amazon SageMaker
-------------------------------------------------------------

With the data now processed you can follow the
`GraphStorm Amazon SageMaker guide <https://github.com/awslabs/graphstorm/tree/main/sagemaker#launch-graph-partitioning-task>`_
to partition your data and run training on AWS.
