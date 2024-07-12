.. _gsprocessing_emr_ec2:

Running distributed jobs on EMR on EC2
======================================

Once the :ref:`distributed processing setup<gsprocessing_distributed_setup>` is complete,
and we have built and pushed an EMR image tagged as ``graphstorm-processing-emr``, we can
set up our execution environment for EMR. If you're not familiar with EMR
we suggest going through its
`introductory documentation <https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-what-is-emr.html>`_
to familiarize yourself with its concepts.

In summary, we will launch an EMR cluster configured in a way that will allow
us to run jobs with executors that use the GSProcessing EMR Docker image,
and then launch our job using ``spark-submit`` from the
cluster's leader node.

Follow EMR set-up
-----------------

To get started with EMR we will need to have an administrative user,
and use it to create the required roles and policies for EMR, as well
as an Amazon EC2 key pair for SSH.
To do so follow the EMR `Setting up Amazon EMR guide
<https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-setting-up.html>`_.

Make note of the SSH key pair you plan to use to access the cluster.

Ensure EMR instance role can access the ECR repository
------------------------------------------------------

To ensure we are able to pull the image from ECR within
the EMR cluster launched, we'll need to allow the
EC2 instance profile used by EMR to read from ECR.
To create these roles we can run the following command using an
administrative user:

.. code-block:: bash

    aws emr create-default-roles

The default EMR on EC2
instance profile would be ``EMR_EC2_DefaultRole``, but if you
are using a different role for the EMR-launched EC2 instances
you should modify the respective role.

The easiest way to do so is to attach the
`AmazonEC2ContainerRegistryReadOnly <https://docs.aws.amazon.com/AmazonECR/latest/userguide/security-iam-awsmanpol.html#security-iam-awsmanpol-AmazonEC2ContainerRegistryReadOnly>`_
policy to the EC2 instance profile, e.g. to
``EMR_EC2_DefaultRole``.

If you only want to allow specific repositories (e.g. only ``graphstorm-processing-emr``) you can also
apply least privilege with attaching the following
inline policy:

.. code-block:: json

    {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:BatchCheckLayerAvailability",
                "ecr:BatchGetImage",
                "ecr:DescribeImages",
                "ecr:DescribeImageScanFindings",
                "ecr:DescribeRepositories",
                "ecr:GetAuthorizationToken",
                "ecr:GetDownloadUrlForLayer",
                "ecr:GetLifecyclePolicy",
                "ecr:GetLifecyclePolicyPreview",
                "ecr:GetRepositoryPolicy",
                "ecr:ListImages",
                "ecr:ListTagsForResource"
            ],
            "Resource": "<ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/graphstorm-processing-emr"
        }
    ]
    }

Launch an AWS EMR cluster with GSProcessing step
--------------------------------------------

Once our roles are set up, that is we have an EMR EC2 instance role,
and a user we can use to launch clusters, we can launch a cluster
configured so it will run a GSProcessing job with the GSProcessing EMR on EC2
Docker image, then terminate. We have tested GSProcessing with EMR 7.0.0 and EMR 6.10.0,
and the instructions should apply for any EMR version ``>6.0.0``.
If you have persistent clusters you want to
use to run GSProcessing, you'd have to modify the EMR Dockerfile
accordingly to use an appropriate EMR image as the source image.

We provide a wrapper script that performs most of the configuration
needed to launch the EMR cluster and submit the GSProcessing job
as an [EMR Step](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-submit-step.html).

For more information on running Spark jobs with custom Docker containers see the EMR
`Configure Docker documentation <https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-plan-docker.html>`_
and how to
`run Spark applications with Docker on Amazon EMR <https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-docker.html>`_.

To launch a GSProcessing job with EMR on EC2 we will use the ``graphstorm-processing/scripts/submit_gsp_emr_step.py`` Python
script that uses ``boto3`` to launch a cluster and the corresponding GSProcessing job as a step.
The script has four required arguments:

* ``--entry-point-s3``: We need to upload the GSProcessing entry point,
  ``graphstorm-processing/graphstorm_processing/distributed_executor.py`` to a location
  on S3 from which our leader instance will be able to read it from.
* ``--gsp-arguments``: Here we pass all the arguments to the entry point as one space-separated
  string. To ensure they are parsed as one string, enclose these in double quotes, e.g.
  ``--gsp-arguments "--input-config gsp-config.json --input-prefix s3://my-bucket/raw-data [...]"``.
* ``--instance-type``: The instance type to use for our cluster. Our script only supports
  a uniform instance types currently.
* ``--instance-count``: Number of worker instances to launch for the cluster.

Run ``python graphstorm-processing/scripts/submit_gsp_emr_step.py --help`` for more optional arguments.

Let's demonstrate how we can launch an EC2 cluster with a GSProcessing step
using the above Python script.

.. code-block:: bash

    INSTANCE_TYPE=m6i.4xlarge
    # INSTANCE_TYPE=m6g.4xlarge # Use for arm64 image
    REGION=us-east-1
    CORE_INSTANCE_COUNT=2
    CLUSTER_NAME="${USER}-gsp-${CORE_INSTANCE_COUNT}x-${INSTANCE_TYPE}"

    # GSProcessing arguments
    MY_BUCKET="enter-your-bucket-name-here"
    REGION="bucket-region" # e.g. us-west-2
    INPUT_PREFIX="s3://${MY_BUCKET}/gsprocessing-input"
    OUTPUT_BUCKET=${MY_BUCKET}
    GRAPH_NAME="small-graph"
    CONFIG_FILE="gconstruct-config.json"
    DO_REPARTITION="true"
    GENERATE_REVERSE="true"


    # We assume this script is saved in the same path as submit_gsp_emr_step.py,
    # that is graphstorm-processing/scripts
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    # Upload the entry point to S3
    PATH_TO_ENTRYPOINT="$SCRIPT_DIR/../graphstorm_processing/distributed_executor.py"
    S3_ENTRY_POINT="s3://${OUTPUT_BUCKET}/emr-scripts/distributed_executor.py"
    aws s3 cp "${PATH_TO_ENTRYPOINT}" ${S3_ENTRY_POINT}

    OUTPUT_PREFIX="s3://${OUTPUT_BUCKET}/gsprocessing/emr/${GRAPH_NAME}"

    python "${SCRIPT_DIR}/submit_gsp_emr_step.py" \
        --entry-point-s3 ${S3_ENTRY_POINT} \
        --instance-type ${INSTANCE_TYPE} \
        --log-uri "${OUTPUT_PREFIX}/spark-logs" \
        --worker-count ${CORE_INSTANCE_COUNT} \
        --gsp-arguments "--config-filename ${CONFIG_FILE} \
            --input-prefix ${INPUT_PREFIX} \
            --output-prefix ${OUTPUT_PREFIX} \
            --add-reverse-edges ${GENERATE_REVERSE} \
            --do-repartition ${REPARTITION_ON_LEADER}"

Running the above will return a cluster ID, which you can use to monitor the
GSProcessing job execution.

We can also run a waiter to wait for the job to finish before checking logs.

.. code-block:: bash

    aws emr wait step-complete --cluster-id j-XXXXXXXXXX --region ${REGION} && echo "GSProcessing job complete."

Ensure row counts are aligned and terminate the cluster
---------------------------------------------------

By setting ``--do-repartition True`` on our job launch script
we have ensured that the row count alignment step will run on the
Spark leader, making the output of GSProcessing ready to be used
with distributed partitioning. To ensure the process completed
successfully, we can run:

.. code-block:: bash

    aws s3 ls ${OUTPUT_PREFIX}

                               PRE edges/
                               PRE node_data/
                               PRE raw_id_mappings/
    2023-08-05 00:47:36        804 launch_arguments.json
    2023-08-05 00:47:36       1916 gconstruct-config.json
    2023-08-05 00:47:36      11914 metadata.json
    2023-08-05 00:47:37        545 perf_counters.json
    2023-08-05 00:47:37      12082 updated_row_counts_metadata.json

We should see the file ``updated_row_counts_metadata.json`` in the output,
which means our data are ready for distributed partitioning.

If the re-partitioning failed, we can run a separate job, see :ref:`row count alignment<row_count_alignment>`
for details.

Run distributed partitioning and training on Amazon SageMaker
-------------------------------------------------------------

With the data now processed you can follow the
`GraphStorm Amazon SageMaker guide
<https://graphstorm.readthedocs.io/en/latest/scale/sagemaker.html#run-graphstorm-on-sagemaker>`_
to partition your data and run training on AWS.
