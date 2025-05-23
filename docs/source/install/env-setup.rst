.. _setup:

Environment Setup
=================

This guide will walk you through the process of installing GraphStorm, based on your specific use scenario.

GraphStorm supports three environment setup methods:
    - Install GraphStorm to your local Python environment. This method is ideal for model development and testing on a single machine.
    - Setup a GraphStorm Docker image. This method allows you to work in a reproducible environment, and
      can naturally be expanded to use GraphStorm in a distributed multi-machine environment.
    - :ref:`Run GraphStorm jobs on SageMaker <distributed-sagemaker>`. This method makes it easy to run
      distributed jobs on massive graphs without worrying about infrastructure setup and management, allowing you to focus on your business problems.

.. _setup_pip:

Setup GraphStorm in your local Python environment
----------------------------------------------------

Prerequisites
...............

1. **Linux OS**: The current version of GraphStorm supports Linux-based operating systems. GraphStorm
has been tested on Ubuntu 20.04, 22.04 and AL2023.

2. **Python3**: The current version of GraphStorm requires Python version **3.8** to **3.11**.

3. (Optional) GraphStorm supports **Nvidia GPUs**.

Install Dependencies
.....................

GraphStorm requires ``PyTorch>=1.13.0`` and ``DGL>=1.1.3``.

We recommend users to install PyTorch v2.3.0 and DGL v2.3.0 for best compatibility.

For users who have to use older DGL versions, please refer to `install GraphStorm with DGL 1.1.3 <https://graphstorm.readthedocs.io/en/v0.4/install/env-setup.html#install-graphstorm>`_.

For Nvidia GPU environment:

.. code-block:: bash

    # for CUDA 11
    pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
    pip install dgl==2.3.0+cu118 -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html

    # for CUDA 12
    pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
    pip install dgl==2.3.0+cu121 -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

For CPU environment:

.. code-block:: bash

    pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
    pip install dgl==2.3.0 -f https://data.dgl.ai/wheels/torch-2.3/repo.html


Install GraphStorm
...................

Users can use ``pip`` or ``pip3`` to install GraphStorm.

.. code-block:: bash

    pip install graphstorm


Clone GraphStorm codebase (Optional)
..........................................
The GraphStorm repository includes a set of scripts, tools, and examples, which can facilitate the use of the
framework.

* **graphstorm/training_scripts/** and **graphstorm/inference_scripts/** include example configuration yaml files that are used in GraphStorm documentations and tutorials and can be used as a starting point for
your own training configuration.
* **graphstorm/examples** includes use-case specific examples, such as temporal graph learning, using SageMaker Pipelines, or performing graph-level predictions with GraphStorm.
* **graphstorm/tools** includes utilities for GraphStorm, such as data sanity checks for partitioned graph data.
* **graphstorm/sagemaker** has fully-fledged launch scripts to help your run GraphStorm jobs on Amazon SageMaker and create and execute SageMaker Pipelines.

You can clone the GraphStorm repository to get access to these tools and examples:

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git

.. _setup_docker:

Setup GraphStorm Docker Environment
-----------------------------------

Running GraphStorm within a Docker container will allow you to have a reproducible environment to run
examples without affecting your local Python environment.

Prerequisites
...............

1. **Docker**: You need to install Docker in your environment following
`Docker documentation <https://docs.docker.com/engine/install/>`_.

Using Docker's convenience script you can install Docker on a Linux machine:

.. code-block:: bash

    sudo apt update
    sudo apt install -y ca-certificates curl
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh ./get-docker.sh --dry-run # Preview the commands
    # Run the installation once ready
    # sudo sh ./get-docker.sh

.. note::

    After installing Docker, you may need to add your user to the docker group to run Docker commands without sudo:

    .. code-block:: bash

        sudo usermod -aG docker $USER
        # Log out and back in for the changes to take effect

2. (Optional) GraphStorm supports **Nvidia GPUs** for GPU-based training and inference. To launch
containers with GPU support you need the `Nvidia Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_. If using AWS `Deep Learning AMI GPU`, the Nvidia Container Toolkit comes preinstalled.

.. _build_docker:

Build a GraphStorm Docker image
...............................

Set up AWS access
-----------------

To build and push the image to the Amazon Elastic Container Registry (ECR) you need the
``aws-cli`` and you will need valid AWS credentials as well.

To `install the AWS CLI <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_
you can use:

.. code-block:: bash

    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install

To set up credentials for use with ``aws-cli`` see the
`AWS docs <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html#cli-configure-files-examples>`_.

Your executing role should have full ECR access to be able to pull from ECR to build the image,
create an ECR repository if it doesn't exist, and push the GraphStorm image to the repository.
See the `official ECR docs <https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-push-iam.html>`_
for details.


Building the GraphStorm images using Docker
-------------------------------------------

With Docker installed, and your AWS credentials set up,
you can use the provided scripts
in the ``graphstorm/docker`` directory to build the image.

GraphStorm supports Amazon SageMaker and EC2/local
execution environments, so you need to choose which image you want
to build first.

The ``build_graphstorm_image.sh`` script can build the image
locally and tag it. It only requires providing the intended execution environment,
using the ``-e/--environment`` argument. The supported environments
are ``sagemaker`` to run jobs on Amazon SageMaker and ``local`` to run jobs
on local instances, like a custom cluster of EC2 instances.

For example, can use the following commands to build the local image
with GPU support:

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git
    cd graphstorm
    bash docker/build_graphstorm_image.sh --environment local

The above will use the local Dockerfile for GraphStorm,
build an image and tag it as ``graphstorm:local-gpu``.

The script also supports other arguments to customize the image name,
tag and other aspects of the build. We list the full argument list below:

* ``-x, --verbose``       Print script debug info (set -x)
* ``-e, --environment``   Image execution environment. Must be one of 'local' or 'sagemaker'. Required.
* ``-d, --device``        Device type, must be one of 'cpu' or 'gpu'. Default is 'gpu'.
* ``-p, --path``          Path to graphstorm root directory, default is one level above the script's location.
* ``-i, --image``         Docker image name, default is 'graphstorm'.
* ``-s, --suffix``        Suffix for the image tag, can be used to push custom image tags. Default is
  "<environment>-<device>", e.g. ``sagemaker-gpu``.
* ``-b, --build``         Docker build directory prefix, default is ``/tmp/graphstorm-build/docker``.
* ``--use-parmetis``      When this flag is set we add the `ParMETIS <https://license.umn.edu/product/parmetis---mesh-graph-partitioning-algorithm>`_
  dependencies to the image. ParMETIS is an advanced distributed graph partitioning algorithm designed
  to minimize communication time during GNN training.

For example you can build an image to support CPU-only execution using:

.. code-block:: bash

    bash docker/build_graphstorm_image.sh --environment local --device cpu
    # Will build an image named 'graphstorm:local-cpu'

Or to build and tag an image to run ParMETIS with EC2 instances:

.. code-block:: bash

    bash docker/build_graphstorm_image.sh --environment local --device cpu --use-parmetis --suffix "-parmetis"
    # Will build an image named 'graphstorm:local-cpu-parmetis'

See ``bash docker/build_graphstorm_image.sh --help``
for more information.

Create a GraphStorm Container
..............................

Once you have built the image, you can launch a local container to run test jobs.

If your host has access to a GPU run the following command:

.. code:: bash

    docker run --gpus all --network=host --rm -v /dev/shm:/dev/shm/ -d --name gs-test graphstorm:local-gpu

Or if using a CPU-only host:

.. code:: bash

    docker run --network=host -v /dev/shm:/dev/shm/ --rm -d --name gs-test graphstorm:local-cpu

This command will create a GraphStorm container, named ``gs-test`` and run the container as a daemon.

.. note::

    Notice that we assign the host's shared memory volume to the container as well using
    ``-v /dev/shm:/dev/shm/``. GraphStorm uses shared memory to host graph data, so it is important
    that you allocate enough shared memory to the container. You can also set the shared memory
    using e.g. ``--shm-size 4gb``.

To connect to the running container use the following command:

.. code:: bash

    docker container exec -it gs-test /bin/bash

If successful, the command prompt will change to the container's, like

.. code-block:: console

    root@<ip-address>:/#

.. note::

    If you are planning to run GraphStorm in a local cluster, specific instruction for running GraphStorm with an NFS shared filesystem is given in :ref:`Use GraphStorm in a Distributed Cluster<distributed-cluster>`.

After exiting (Ctrl+D) you can stop the container using

.. code:: bash

    docker container kill gs-test


Push the image to Amazon Elastic Container Registry (ECR)
---------------------------------------------------------

Once you build the image, you can use the ``push_graphstorm_image.sh`` script to push the image
to an `Amazon ECR <https://docs.aws.amazon.com/AmazonECR/latest/userguide/what-is-ecr.html>`_ repository.
ECR allows you to easily store, manage, and deploy container images.

This will allow you to use the image in SageMaker jobs using SageMaker Bring-Your-Own-Container, or to launch
EC2 clusters.

The script requires you to provide the intended execution environment again using
the ``-e/--environment`` argument,
and by default will create a repository named ``graphstorm`` in the ``us-east-1`` region,
on the default AWS account ``aws-cli`` is configured for,
and push the image tagged as ``<environment>-<device>``.
The script will try to create a new ECR repository if one doesn't already exist.

In addition to ``-e/--environment``, the script supports several optional arguments, for a full list use
``bash push_graphstorm_image.sh --help``. We list the most important below:

* ``-e, --environment``   Image execution environment. Must be one of 'local' or 'sagemaker'. Required.
* ``-a, --account``       AWS Account ID to use, we try retrieve the default from the AWS CLI configuration.
* ``-r, --region``        AWS Region to push the image to, we retrieve the default from the AWS CLI configuration.
* ``-d, --device``        Device type, must be one of 'cpu' or 'gpu'. Default is 'gpu'.
* ``-p, --path``          Path to graphstorm root directory, default is one level above the script's location.
* ``-i, --image``         Docker image name, default is 'graphstorm'.
* ``-s, --suffix``        Suffix for the image tag, can be used to push custom image tags. Default is "<environment>-<device>", e.g. ``sagemaker-gpu``.
* ``-x, --verbose``       Print script debug info (set -x)

Examples:

.. code-block:: bash

    # Push an image to '123456789012.dkr.ecr.us-east-1.amazonaws.com/graphstorm:local-cpu'
    bash docker/push_graphstorm_image.sh -e local -r "us-east-1" -a "123456789012" --device cpu
    # Push the ParMETIS-capable image you previously built to '123456789012.dkr.ecr.us-east-1.amazonaws.com/graphstorm:local-cpu-parmetis'
    bash docker/push_graphstorm_image.sh -e local -r "us-east-1" -a "123456789012" --device cpu --suffix "-parmetis"
