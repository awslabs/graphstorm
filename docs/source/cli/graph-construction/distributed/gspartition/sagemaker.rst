==========================================
Running partition jobs on Amazon SageMaker
==========================================

Once the :ref:`distributed processing<gsprocessing_distributed_setup>` is complete,
you can use Amazon SageMaker launch scripts to launch distributed processing jobs with AWS resources.

Build the Docker Image for GSPartition Jobs on Amazon SageMaker
---------------------------------------------------------------
GSPartition job on Amazon SageMaker uses its SageMaker's **BYOC** (Bring Your Own Container) mode.

To build and push the GraphStorm SageMaker image follow the instructions
in :ref:`Setup GraphStorm SageMaker Docker Image<build_sagemaker_docker>`.

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
       --num-parts 2 \
       --instance-count 2 \
       --output-data-s3 ${OUTPUT_PATH} \
       --instance-type ml.t3.xlarge \
       --image-url ${IMAGE_URI} \
       --region ${REGION} \
       --role ${ROLE}  \
       --entry-point "run/partition_entry.py" \
       --metadata-filename ${METADATA_FILE} \
       --log-level INFO \
       --partition-algorithm random

.. warning::
    The ``--num-parts`` should be equal to the ``--instance-count`` here.

Running the above will take the dataset after GSProcessing
from ``${DATASET_S3_PATH}`` as input and create a DistDGL graph with
``${NUM_PARTITIONS}`` under the output path, ``${OUTPUT_PATH}``.
Currently we only support ``random`` as the partitioning algorithm for sagemaker.
