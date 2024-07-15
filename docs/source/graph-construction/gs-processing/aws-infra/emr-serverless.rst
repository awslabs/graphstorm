.. _gsprocessing_emr_serverless:

Running distributed jobs on EMR Serverless
==========================================

Once the :ref:`distributed processing setup<gsprocessing_distributed_setup>` is complete,
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
docker image: :ref:`GSProcessing Distributed Setup<gsprocessing_distributed_setup>` to save cost and time. Please note that the maximum size for docker images in EMR Serverless is limited to 5GB:
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

To launch the same example job as we demonstrate in the :ref:`SageMaker Processing job guide<gsprocessing_sagemaker>`
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
launch a SageMaker job as below after following the :ref:`distributed processing setup<gsprocessing_distributed_setup>`
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
:ref:`row count alignment<row_count_alignment>`.

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


Run distributed partitioning and training on Amazon SageMaker
-------------------------------------------------------------

With the data now processed you can follow the
`GraphStorm Amazon SageMaker guide
<https://graphstorm.readthedocs.io/en/latest/scale/sagemaker.html#run-graphstorm-on-sagemaker>`_
to partition your data and run training on AWS.
