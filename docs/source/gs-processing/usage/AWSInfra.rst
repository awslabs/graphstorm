================================================
Running distributed processing jobs on AWS Infra
================================================

This doc illustrates different usage of GSProcessing on different AWS Infras.
Followings are for quick reference for different AWS Infras:

Running distributed jobs on Amazon SageMaker: :ref:`sagemaker`.

Running distributed jobs on EMR Serverless: :ref:`emr_serverless`

Running distributed jobs on EMR on EC2: :ref:`emr_ec2`

.. _sagemaker:

Running distributed jobs on Amazon SageMaker
============================================

Once the :doc:`Amazon SageMaker setup <distributed-processing-setup>` is complete, we can
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
For more details on the re-partitioning step see :doc:`row-count-alignment`.

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

.. _emr_serverless:

Running distributed jobs on EMR Serverless
==========================================

Once the :doc:`distributed processing setup <distributed-processing-setup>` is complete,
and we have built and pushed an EMR Serverless image tagged as ``graphstorm-processing-emr-serverless``, we can
set up our execution environment for EMR Serverless (EMR-S). If you're not familiar with EMR-S
we suggest going through its `introductory documentation <https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/emr-serverless.html>`_
to familiarize yourself with its concepts.

In summary, we will set up an EMR-S `Application`, which we will configure to use our EMR-S
image, and then we'll demonstrate how we can launch jobs using the EMR-S application we created.

.. note::

    Because the set-up of EMR-S involves role creation and modifying the permissions of our ECR repository,
    we will need access to a role with IAM access, usually an administrative role.

Follow EMR Serverless set-up
----------------------------

To get started with EMR-S we will need to have an administrative user,
and use it to create the required roles and policies for EMR-S.
To do so follow the EMR-S `Setting up guide
<https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/setting-up.html>`_.

Create a job runtime role for EMR Serverless
---------------------------------------------

To be able to run EMR-S jobs we will need access to a role that
is configured with access to the S3 bucket we will use.

Follow the `Create a job runtime role
<https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/getting-started.html#gs-prerequisites>`_
guide to create such a role. You can replace ``DOC-EXAMPLE-BUCKET`` with the bucket you used
to upload your test data in :ref:`gsp-upload-data-ref`.

Ensure EMR-S service role can access the ECR repository
-------------------------------------------------------

To ensure we can create EMR-S applications and run jobs
using our custom image, we need to give the EMR-S service
role the ability to pull the image from our ECR repository.

To do so we need to add ECR actions to the entity that
creates the EMR-S applications, and configure our ECR
repository to provide access to our
EMR-S application.

To ensure the entity that creates the EMR-S application
can perform ECR actions, follow the
`Prerequisites <https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/application-custom-image.html#worker-configs>`_
part of the `Customizing an image` EMR-S guide. If you're using
an administrative user to work through this process you might
already have full ECR access.

If not using an administrative user, the relevant policy to attach to the role/user
you are using would be:

.. code-block:: json

    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ECRRepositoryListGetPolicy",
                "Effect": "Allow",
                "Action": [
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:DescribeImages"
                ],
                "Resource": "<ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/graphstorm-processing-emr-serverless"
            }
        ]
    }

Create an EMR-S application that uses our custom image
------------------------------------------------------

Next we will need to create an EMR-S application that
uses our custom image.
For a general guide see the
`official docs <https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/application-custom-image.html#create-app>`_.

Here we will just show the custom image application creation using the AWS CLI:

.. code-block:: bash

    aws emr-serverless create-application \
        --name gsprocessing-0.2.2 \
        --release-label emr-6.13.0 \
        --type SPARK \
        --image-configuration '{
            "imageUri": "<aws-account-id>.dkr.ecr.<region>.amazonaws.com/graphstorm-processing-emr-serverless:0.2.2-<arch>"
        }'

Here you will need to replace ``<aws-account-id>``, ``<arch>`` (``x86_64`` or ``arm64``), and ``<region>`` with the correct values
from the image you just created. GSProcessing version ``0.2.2`` uses ``emr-6.13.0`` as its
base image, so we need to ensure our application uses the same release.

Additionally, if it is required to use text feature transformation with Huggingface model, it is suggested to download the model cache inside the emr-serverless
docker image: :doc:`distributed-processing-setup` to save cost and time. Please note that the maximum size for docker images in EMR Serverless is limited to 5GB:
`EMR Serverless Considerations and Limitations
<https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/application-custom-image.html#considerations>`_.



Allow EMR Serverless to access the custom image repository
----------------------------------------------------------

Finally we need to provide the EMR-S service Principal access
to the `graphstorm-processing-emr-serverless` ECR image repository,
for which we will need to modify the repository's policy statement.

As shown in the
`EMR docs <https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/application-custom-image.html#access-repo>`_,
once we have the EMR-S Application ID (from creating the application in the previous step)
we can use it to limit access to the repository to that particular application.

The policy we need to set would be the following:

.. code-block:: json

    {
        "Version": "2012-10-17",
        "Statement": [
            {
            "Sid": "Emr Serverless Custom Image Support",
            "Effect": "Allow",
            "Principal": {
                "Service": "emr-serverless.amazonaws.com"
            },
            "Action": [
                "ecr:BatchGetImage",
                "ecr:DescribeImages",
                "ecr:GetDownloadUrlForLayer"
            ],
            "Condition":{
                "StringEquals":{
                "aws:SourceArn": "arn:aws:emr-serverless:<region>:<aws-account-id>:/applications/<application-id>"
                }
            }
            }
        ]
    }

Where you would need to replace values for ``<aws-account-id>``, ``<region>``, and ``<application-id>``.

See `Setting a private repository policy statement <https://docs.aws.amazon.com/AmazonECR/latest/userguide/set-repository-policy.html>`_
for how to set a repository policy.


Running GSProcessing jobs on EMR Serverless
-------------------------------------------

With all the setup complete we should now have the following:

* An ECR repository where we have pushed the GSProcessing EMR-S image,
  and to which we have provided access to the EMR-S application we just created.
* An EMR-S application that uses our custom image.
* An execution role that our EMR-S jobs will use when we launch them.

To launch the same example job as we demonstrate in the :doc:`SageMaker Processing job guide <amazon-sagemaker>`
you can use the following ``bash`` snippet. Note that we use ``jq`` to wrangle JSON data,
which you can download from its `official website <https://jqlang.github.io/jq/download/>`_,
install using your package manager, or by running ``pip install jq``.

Before starting  the job, make sure you have uploaded the input data
as described in :ref:`gsp-upload-data-ref`.

.. code-block:: bash

    APPLICATION_ID="enter-your-application-id-here"
    ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
    MY_BUCKET="enter-your-bucket-name-here"
    EMR_S_ROLE_NAME="enter-your-emr-serverless-execution-role-name-here"
    REGION="bucket-region" # e.g. us-west-2
    INPUT_PREFIX="s3://${MY_BUCKET}/gsprocessing-input"
    OUTPUT_BUCKET=${MY_BUCKET}
    GRAPH_NAME="small-graph"
    CONFIG_FILE="gconstruct-config.json"
    NUM_FILES="-1"
    DO_REPARTITION="true"
    GSP_HOME="enter/path/to/graphstorm/graphstorm-processing/"

    LOCAL_ENTRY_POINT=$GSP_HOME/graphstorm_processing/distributed_executor.py
    S3_ENTRY_POINT="s3://${OUTPUT_BUCKET}/emr-serverless-scripts/distributed_executor.py"

    ROLE="arn:aws:iam::${ACCOUNT}:role/${EMR_S_ROLE_NAME}"

    export OUTPUT_PREFIX="s3://${OUTPUT_BUCKET}/gsprocessing/emr-s/${GRAPH_NAME}/${NUM_FILES}files/"

    # Copy entry point script to S3 to ensure latest version is used
    aws s3 cp $LOCAL_ENTRY_POINT $S3_ENTRY_POINT

    # Construct arguments JSON string using jq
    ARGS_JSON=$( jq -n \
        --arg entry "$S3_ENTRY_POINT" \
        --arg in "$INPUT_PREFIX" \
        --arg out "$OUTPUT_PREFIX" \
        --arg cfg "$CONFIG_FILE" \
        --arg nfiles "$NUM_FILES" \
        --arg gname "$GRAPH_NAME" \
        --arg repart "$DO_REPARTITION" \
        '{
            sparkSubmit: {
                entryPoint: $entry,
                entryPointArguments:
                    ["--input-prefix", $in,
                    "--output-prefix", $out,
                    "--config-file", $cfg,
                    "--num-output-files", $nfiles,
                    "--graph-name", $gname,
                    "--do-repartition", $repart]
            }
        }' )

    echo "Arguments JSON:"
    echo $ARGS_JSON | jq -r

    echo "Starting EMR-S job..."
    aws --region $REGION emr-serverless start-job-run \
        --name "gsprocessing-emr-s-example" \
        --application-id $APPLICATION_ID \
        --execution-role-arn $ROLE \
        --job-driver "${ARGS_JSON}" # Need to surround ARGS_JSON with quotes here to maintain JSON formatting

Running the re-partition job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to the SageMaker example, we set the ``do-repartition`` value to ``True``,  to try to re-partition our
data on the Spark leader. If the data are too large to re-partition on the Spark leader,
we need to run a follow-up job to align the output with the
expectations of the DistDGL partitioning pipeline. The easiest is to run the job locally
on an instance with S3 access (where we installed GSProcessing):

.. code-block:: bash

    gs-repartition --input-prefix ${OUTPUT_PREFIX}

Or if your data are too large for the re-partitioning job to run locally, you can
launch a SageMaker job as below after following the :doc:`distributed processing setup <distributed-processing-setup>`
and building the GSProcessing SageMaker ECR image:

.. code-block:: bash

    bash docker/build_gsprocessing_image.sh --environment sagemaker --region ${REGION}
    bash docker/push_gsprocessing_image.sh --environment sagemaker --region ${REGION}

    SAGEMAKER_ROLE_NAME="enter-your-sagemaker-execution-role-name-here"
    IMAGE_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/graphstorm-processing-sagemaker:latest-x86_64"
    ROLE="arn:aws:iam::${ACCOUNT}:role/service-role/${SAGEMAKER_ROLE_NAME}"
    INSTANCE_TYPE="ml.t3.xlarge"

    python scripts/run_repartitioning.py --s3-input-prefix ${OUTPUT_PREFIX} \
        --role ${ROLE} --image ${IMAGE_URI}  --config-filename "metadata.json" \
        --instance-type ${INSTANCE_TYPE} --wait-for-job


Note that ``${OUTPUT_PREFIX}`` here will need to match the value assigned when launching
the EMR-S job, i.e. ``"s3://${OUTPUT_BUCKET}/gsprocessing/emr-s/small-graph/4files/"``

For more details on the re-partitioning step see
:doc:`row-count-alignment`.

Examine the output
------------------

Once both the jobs are finished we can examine the output created, which
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

.. _emr_ec2:

Running distributed jobs on EMR on EC2
======================================

Once the :doc:`distributed processing setup <distributed-processing-setup>` is complete,
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

If the re-partitioning failed, we can run a separate job, see :doc:`row-count-alignment`
for details.

Run distributed partitioning and training on Amazon SageMaker
=============================================================

With the data now processed you can follow the
`GraphStorm Amazon SageMaker guide
<https://graphstorm.readthedocs.io/en/latest/scale/sagemaker.html#run-graphstorm-on-sagemaker>`_
to partition your data and run training on AWS.
