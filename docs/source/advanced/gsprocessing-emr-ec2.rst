.. _gsprocessing_emr_ec2_customized_clusters:

Running distributed graph processing on customized EMR-on-EC2 clusters
=======================================================================

For advanced users who want to launch GSProcessing job through ``spark-submit``
from the cluster's leader node, :ref:`GSProcessing <gsprocessing_distributed_setup>`
provides another customized solution to setup
`EMR clusters <https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-plan-ec2-instances.html>`_.
:ref:`distributed processing setup<gsprocessing_emr_ec2>`

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

Create a security group that will allow us to SSH to the leader instance
------------------------------------------------------------------------

In order to be able to launch Spark jobs from within the leader instance
we will need to create an EC2 security group that will allow us to login
to the EMR leader.

To do so follow the `AWS docs <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/authorizing-access-to-an-instance.html#add-rule-authorize-access>`_,
and make note of the security group ID, e.g `sg-XXXXXXXXXXXXXXXXX`.

Launch an EMR cluster with the appropriate permissions
------------------------------------------------------

Once our roles are set up, that is we have an EMR EC2 instance role,
and a user we can use to launch clusters, we can launch a cluster
configured to allow us to run jobs with the GSProcessing EMR on EC2
Docker image. We have tested GSProcessing with EMR 7.0.0 and EMR 6.10.0,
and the instructions should apply for any EMR version ``>6.0.0``.
If you have persistent clusters you want to
use to run GSProcessing, you'd have to modify the EMR Dockerfile
accordingly to use an appropriate EMR image as the source image.

When launching the cluster, we need to provide a configuration to the launch
command to trust the GSProcessing ECR repository:

.. code-block:: json

    [
    {
        "Classification": "container-executor",
        "Configurations": [
            {
                "Classification": "docker",
                "Properties": {
                    "docker.trusted.registries": "local,centos,<ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/graphstorm-processing-emr",
                    "docker.privileged-containers.registries": "local,centos,<ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/graphstorm-processing-emr"
                }
            }
        ]
    }
    ]

Here you would replace the placeholder values for ``<ACCOUNT>`` and ``<REGION>``
with the appropriate values  for your account. Save this
script and name it `container-executor.json`, we'll use it in the next step.

For more information on running Spark jobs with custom Docker containers see the EMR
`Configure Docker documentation <https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-plan-docker.html>`_
and how to
`run Spark applications with Docker on Amazon EMR <https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-docker.html>`_.

To launch an EMR cluster using the AWS CLI we can use a ``bash`` script like the following.

.. code-block:: bash

    KEYPAIR=my-key-pair-name
    SUBNET_ID=subnet-XXXXXXXX
    MASTER_SG=sg-XXXXXXXXXXXXXXXXX # Use the security group with ssh access
    INSTANCE_TYPE=m6i.4xlarge
    # INSTANCE_TYPE=m6g.4xlarge # Use for arm64 image
    REGION=us-east-1
    EMR_VERSION="emr-7.0.0"
    CORE_INSTANCE_COUNT=3
    CLUSTER_NAME="${USER}-gsp-${CORE_INSTANCE_COUNT}x-${INSTANCE_TYPE}"
    INSTANCE_ROLE="EMR_EC2_DefaultRole"
    TERMINATION_HOURS=1

    # We assume this script is saved in the same path as container-executor.json
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

    LOG_BUCKET=my-log-bucket

    aws emr create-cluster \
        --applications Name=Hadoop Name=Spark \
        --auto-termination-policy IdleTimeout=$((${TERMINATION_HOURS}*60*60)) \
        --configurations file://${SCRIPT_DIR}/container-executor.json \
        --ec2-attributes KeyName=${KEYPAIR},SubnetId=${SUBNET_ID},AdditionalMasterSecurityGroups=${MASTER_SG} \
        --instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=${INSTANCE_TYPE} \
            InstanceGroupType=CORE,InstanceCount=${CORE_INSTANCE_COUNT},InstanceType=${INSTANCE_TYPE} \
        --log-uri s3://${LOG_BUCKET}/emr-logs/ \
        --name ${CLUSTER_NAME} \
        --region ${REGION} \
        --release-label ${EMR_VERSION} \
        --use-default-roles

Running the above will return a JSON structure like:

.. code-block:: json

    {
        "ClusterId": "j-XXXXXXXXXX",
        "ClusterArn": "arn:aws:elasticmapreduce:us-east-1:<ACCOUNT>:cluster/j-XXXXXXXXXX"
    }

Make note of the cluster ID, which we will use to log into the leader instance.

We can also run a waiter to ensure we only proceed to the next step when the cluster is
ready to run jobs:

.. code-block:: bash

    aws emr wait cluster-running --cluster-id j-XXXXXXXXXX --region ${REGION} && echo "Cluster ready"

Log in to the leader and submit a GSProcessing job
--------------------------------------------------

To submit a job we can use a helper ``bash`` script, which we list below:

.. code-block:: bash

    # submit-gsp-job.sh
    #!/usr/bin/env bash
    set -euox pipefail

    MY_BUCKET="enter-your-bucket-name-here"
    REGION="bucket-region" # e.g. us-west-2
    INPUT_PREFIX="s3://${MY_BUCKET}/gsprocessing-input"
    NUM_EXECUTORS=2
    OUTPUT_BUCKET=${MY_BUCKET}
    GRAPH_NAME="small-graph"
    CONFIG_FILE="gconstruct-config.json"

    ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

    REPOSITORY="graphstorm-processing-emr"
    ARCH="x86_64"
    TAG="latest-${ARCH}"
    IMAGE="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY}:${TAG}"

    S3_ENTRY_POINT="s3://${OUTPUT_BUCKET}/emr-scripts/distributed_executor.py"

    export OUTPUT_PREFIX="s3://${OUTPUT_BUCKET}/gsprocessing/emr/${GRAPH_NAME}/"

    spark-submit --master yarn \
        --deploy-mode cluster \
        --conf spark.executorEnv.YARN_CONTAINER_RUNTIME_TYPE=docker \
        --conf spark.executorEnv.YARN_CONTAINER_RUNTIME_DOCKER_IMAGE=${IMAGE} \
        --conf spark.executorEnv.PYSPARK_PYTHON="/.pyenv/shims/python" \
        --conf spark.yarn.appMasterEnv.YARN_CONTAINER_RUNTIME_TYPE=docker \
        --conf spark.yarn.appMasterEnv.YARN_CONTAINER_RUNTIME_DOCKER_IMAGE=${IMAGE} \
        --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON="/.pyenv/shims/python" \
        --num-executors ${NUM_EXECUTORS} \
        ${S3_ENTRY_POINT} \
            --config-filename ${CONFIG_FILENAME} \
            --input-prefix ${INPUT_PREFIX} \
            --output-prefix ${OUTPUT_PREFIX} \
            --do-repartition True


We will need to save and upload this helper script to the Spark leader,
and the ``distributed_executor.py`` entry point to an S3 location that the leader can access.
From where you cloned graphstorm you can run:

.. code-block:: bash

    MY_BUCKET="enter-your-bucket-name-here" # The leader instance needs to be able to read this bucket
    aws s3 cp /path/to/graphstorm/graphstorm-processing/graphstorm_processing/distributed_executor.py
        \ "s3://${MY_BUCKET}/emr-scripts/distributed_executor.py"
    aws emr put --cluster-id j-XXXXXXXXXX --key-pair-file /path/to/my-key-pair.pem \
        --src submit-gsp-job.sh

Once the cluster is launched we can use the key pair
we created and the cluster ID to log into the Spark leader
to submit jobs. We can do so by running:

.. code-block:: bash

    aws emr ssh --cluster-id j-XXXXXXXXXX --key-pair-file /path/to/my-key-pair.pem \
        --region ${REGION}

    bash submit-gsp-job.sh

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

Once done, remember to clean up your cluster resources by terminating the cluster:

.. code-block:: bash

    aws emr terminate-clusters --cluster-ids j-XXXXXXXXXX

Run distributed partitioning and training on Amazon SageMaker
-------------------------------------------------------------

With the data now processed you can follow the
`GraphStorm Amazon SageMaker guide
<https://graphstorm.readthedocs.io/en/latest/scale/sagemaker.html#run-graphstorm-on-sagemaker>`_
to partition your data and run training on AWS.
