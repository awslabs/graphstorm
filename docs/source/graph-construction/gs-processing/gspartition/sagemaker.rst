==========================================
Running partition jobs on Amazon SageMaker
==========================================

Once the :ref:`distributed processing<gsprocessing_distributed_setup>` is complete,
you can use Amazon SageMaker launch scripts to launch distributed processing jobs with AWS resources.

Build the Docker Image for GSPartition Jobs on Amazon SageMaker
---------------------------------------------------------------
GSPartition job on Amazon SageMaker uses its SageMaker's **BYOC** (Bring Your Own Container) mode.

Step 1: Build an Amazon SageMaker-compatible Docker image
..........................................................

.. note::
    * Please make sure your account has access key (AK) and security access key (SK) configured to authenticate accesses to AWS services, users can refer to `example policy <https://docs.aws.amazon.com/AmazonECR/latest/userguide/security_iam_id-based-policy-examples.html#security_iam_id-based-policy-examples-access-one-bucket>`_.
    * For more details of Amazon ECR operation via CLI, users can refer to the `Using Amazon ECR with the AWS CLI document <https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html>`_.

First, in a Linux machine, configure a Docker environment by following the `Docker documentation <https://docs.docker.com/get-docker/>`_ suggestions.

In order to use the Amazon SageMaker base Docker image, users need to refer the `DLC image command <https://github.com/aws/deep-learning-containers/blob/master/available_images.md>`_
to find the specific Docker image commands. For example, below is the command for user authentication to access the Amazon SageMaker base Docker image.

.. code-block:: bash

    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

.. note::
    For region other than ``us-east-1``, please refer to `available region <https://docs.aws.amazon.com/sagemaker/latest/dg-ecr-paths/sagemaker-algo-docker-registry-paths.html>`_

Secondly, clone GraphStorm source code, and build a GraphStorm SageMaker compatible Docker image from source with commands:

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git

    cd /path-to-graphstorm/docker/

    bash build_docker_sagemaker.sh ../ <DEVICE_TYPE> <IMAGE_NAME> <IMAGE_TAG>

The ``build_docker_sagemaker.sh`` script takes four arguments:

1. **path-to-graphstorm** (**required**), is the path of the ``graphstorm`` folder, where you cloned the GraphStorm source code. For example, the path could be ``/code/graphstorm``.
2. **DEVICE_TYPE**, is the intended device type of the to-be built Docker image. Please specify ``cpu`` for building CPU-compatible images for partition job.
3. **IMAGE_NAME** (optional), is the assigned name of the to-be built Docker image. Default is ``graphstorm``.

.. warning::
    In order to upload the GraphStorm SageMaker Docker image to Amazon ECR, users need to define the <IMAGE_NAME> to include the ECR URI string, **<AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/**, e.g., ``888888888888.dkr.ecr.us-east-1.amazonaws.com/graphstorm``.

4. **IMAGE_TAG** (optional), is the assigned tag name of the to-be built Docker image. Default is ``sm-<DEVICE_TYPE>``,
   that is, ``sm-cpu`` for CPU images.

Once the ``build_docker_sagemaker.sh`` command completes successfully, there will be a Docker image, named ``<IMAGE_NAME>:<IMAGE_TAG>``,
such as ``888888888888.dkr.ecr.us-east-1.amazonaws.com/graphstorm:sm-cpu``, in the local repository, which could be listed by running:

.. code-block:: bash

    docker images graphstorm

.. _upload_sagemaker_docker:

Step 2: Upload Docker images to Amazon ECR repository
.......................................................
Because Amazon SageMaker relies on Amazon ECR to access customers' own Docker images, users need to upload the Docker images built in the Step 1 to their own ECR repository.

The following command will authenticate the user account to access to user's ECR repository via AWS CLI.

.. code-block:: bash

    aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com

Please replace the `<REGION>` and `<AWS_ACCOUNT_ID>` with your own account information and be consistent with the values used in the **Step 1**.

In addition, users need to create an ECR repository at the specified `<REGION>` with the name as `<IMAGE_NAME>` **WITHOUT** the ECR URI string, e.g., ``graphstorm``.

And then use the following command to push the built GraphStorm Docker image to users' own ECR repository.

.. code-block:: bash

    docker push <IMAGE_NAME>:<IMAGE_TAG>

Please replace the `<IMAGE_NAME>` and `<IMAGE_TAG>` with the actual Docker image name and tag, e.g., ``888888888888.dkr.ecr.us-east-1.amazonaws.com/graphstorm:sm-gpu``.

Launch the GSPartition Job on Amazon SageMaker
-----------------------------------------------
For this example, we'll use an Amazon SageMaker cluster with 2 ``ml.t3.xlarge`` instances.
We assume the data is already on an AWS S3 bucket.
For large graphs, users can choose larger instances or more instances.

Install dependencies
.....................
To run GraphStorm with the Amazon SageMaker service, users should install the Amazon SageMaker library and copy GraphStorm's SageMaker tools.

1. Use the below command to install Amazon SageMaker.

.. code-block:: bash

    pip install sagemaker

2. Copy GraphStorm SageMaker tools. Users can clone the GraphStorm repository using the following command or copy the `sagemaker folder <https://github.com/awslabs/graphstorm/tree/main/sagemaker>`_ to the instance.

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git

Launch GSPartition task
........................
Users can use the following command to launch partition jobs.

.. code:: bash

   python launch/launch_partition.py \
       --graph-data-s3 ${DATASET_S3_PATH} \
       --num-parts ${NUM_PARTITIONS} \
       --instance-count ${NUM_INSTANCES} \
       --output-data-s3 ${OUTPUT_PATH} \
       --instance-type ${INSTANCE_TYPE} \
       --image-url ${IMAGE_URI} \
       --region ${REGION} \
       --role ${ROLE}  \
       --entry-point "run/partition_entry.py" \
       --metadata-filename ${METADATA_FILE} \
       --log-level INFO \
       --partition-algorithm ${ALGORITHM}

.. warning::
    The ``NUM_INSTANCES`` should be equal to the ``NUM_PARTITIONS`` here.

Running the above will take the dataset after GSProcessing
from ``${DATASET_S3_PATH}`` as input and create a DistDGL graph with
``${NUM_PARTITIONS}`` under the output path, ``${OUTPUT_PATH}``.
Currently we only support ``random`` as the partitioning algorithm for sagemaker.


