.. _setup:

Environment Setup
======================
GraphStorm supports two environment setup methods:
    - Install GraphStorm as a pip package. This method works well for development and test on a single machine.
    - Setup a GraphStorm Docker image. This method is good for using GraphStorm in distributed environments that commonly used in production.

.. _setup_pip:

1. Setup GraphStorm with pip Packages
--------------------------------------
Prerequisites
...............

1. **Linux OS**: The current version of GraphStorm supports Linux as the Operation System. We tested GraphStorm on both Ubuntu (22.04 or later version) and Amazon Linux 2.

2. **Python3**: The current version of GraphStorm requires Python installed with the version larger than **3.8**.

3. (Optional) GraphStorm supports **Nvidia GPUs** for using GraphStorm in GPU environments.

Install GraphStorm
...................
Users can use ``pip`` or ``pip3`` to install GraphStorm.

.. code-block:: bash

    pip install graphstorm

Install Dependencies
.....................
Users should install PyTorch v2.1.0 and DGL v1.1.3 that is the core dependency of GraphStorm using the following commands.

For Nvidia GPU environment:

.. code-block:: bash

    # for CUDA 11
    pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install dgl==1.1.3+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html

    # for CUDA 12
    pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install dgl==1.1.3+cu121 -f https://data.dgl.ai/wheels/cu121/repo.html

For CPU environment:

.. code-block:: bash

    pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install dgl==1.1.3 -f https://data.dgl.ai/wheels-internal/repo.html

Configure SSH No-password login (optional)
..........................................
To perform distributed training in a cluster of machines, please use the following commands
to configure a local SSH no-password login that GraphStorm relies on.

.. note::

    The "SSH No-password login" is **NOT** needed for GraphStorm's Standalone mode, i.e., running GraphStorm in one machine only.

.. code-block:: bash

    ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ''
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

Then use this command to test if the SSH no-password login works.

.. code-block:: bash

    ssh 127.0.0.1

If everything is right, the above command will enter another Linux shell process. Then exit this new shell with the command ``exit``.

Clone GraphStorm Toolkits (Optional)
..........................................
GraphStorm provides a set of toolkits, including scripts, tools, and examples, which can facilitate the use of GraphStrom.

* **graphstorm/training_scripts/** and **graphstorm/inference_scripts/** include examplar configuration yaml files that used in GraphStorm documentations and tutorials.
* **graphstorm/examples** includes Python code for customized models and customized data preparation.
* **graphstorm/tools** includes graph partition and related Python code.
* **graphstorm/sagemaker** include commands and code to run GraphStorm on Amazon SageMaker.

Users can clone GraphStorm source code to obtain these toolkits.

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git

.. _setup_docker:

2. Setup GraphStorm Docker Environment
---------------------------------------
Prerequisites
...............

1. **Docker**: You need to install Docker in your environment as the `Docker documentation <https://docs.docker.com/get-docker/>`_ suggests, and the `Nvidia Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_.

For example, in an AWS EC2 instance without Docker preinstalled, you can run the following commands to install Docker.

.. code-block:: bash

    sudo apt-get update
    sudo apt update
    sudo apt install Docker.io

If using AWS `Deep Learning AMI GPU version`, the Nvidia Container Toolkit has been preinstalled.

2. (Optional) GraphStorm supports **Nvidia GPUs** for using GraphStorm in GPU environments.

.. _build_docker:

Build a GraphStorm Docker image from source code
.................................................

Set up AWS access
-----------------

To build and push the image to ECR we'll make use of the
``aws-cli`` and we'll need valid AWS credentials as well.

To install the AWS CLI you can use:

.. code-block:: bash

    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install

To set up credentials for use with ``aws-cli`` see the
`AWS docs <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html#cli-configure-files-examples>`_.

Your executing role should have full ECR access to be able to pull from ECR to build the image,
create an ECR repository if it doesn't exist, and push the GSProcessing image to the repository.
See the [official ECR docs](https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-push-iam.html)
for details.


Building the GraphStorm images using Docker
-------------------------------------------

With Docker installed, and your AWS credentials set up,
you can use the provided scripts
in the ``graphstorm/docker`` directory to build the image.

GraphStorm supports Amazon SageMaker and EC2/local
execution environments, so we need to choose which image we want
to build first.

The ``build_graphstorm_image.sh`` script can build the image
locally and tag it. It only requires providing the intended execution environment,
using the ``-e/--environment`` argument. The supported environments
are ``sagemaker`` and ``local``.

For example, assuming our current directory is where
we cloned ``graphstorm/``, we can use
the following command to build the local image:

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
* ``-s, --suffix``        Suffix for the image tag, can be used to push custom image tags. Default is "<environment>-<device>".
* ``-b, --build``         Docker build directory prefix, default is '/tmp/graphstorm-build/docker'.

For example you can build an image to support CPU-only execution using:

.. code-block:: bash

    bash docker/build_graphstorm_image.sh --environment local --device cpu
    # Will build an image named 'graphstorm:local-cpu'

See ``bash docker/build_graphstorm_image.sh --help``
for more information.

Push the image to Amazon Elastic Container Registry (ECR)
-------------------------------------------------------------

Once the image is built we can use the ``push_graphstorm_image.sh`` script to push the image we just built.
The script will create an ECR repository if needed.

The script again requires us to provide the intended execution environment using
the ``-e/--environment`` argument,
and by default will create a repository named ``graphstorm`` in the ``us-east-1`` region,
on the default AWS account ``aws-cli`` is configured for,
and push the image tagged as ``<environment>-<device>```.

In addition to ``-e/--environment``, the script supports several optional arguments, for a full list use
``bash push_graphstorm_image.sh --help``. We list the most important below:

* ``-e, --environment``   Image execution environment. Must be one of 'local' or 'sagemaker'. Required.
* ``-a, --account``       AWS Account ID to use, we retrieve the default from the AWS cli configuration.
* ``-r, --region``        AWS Region to push the image to, we retrieve the default from the AWS cli configuration.
* ``-d, --device``        Device type, must be one of 'cpu' or 'gpu'. Default is 'gpu'.
* ``-p, --path``          Path to graphstorm root directory, default is one level above the script's location.
* ``-i, --image``         Docker image name, default is 'graphstorm'.
* ``-s, --suffix``        Suffix for the image tag, can be used to push custom image tags. Default is "<environment>-<device>".


Example:

.. code-block:: bash

    bash docker/push_graphstorm_image.sh -e local -r "us-east-1" -a "123456789012"
    # Will push an image to '123456789012.dkr.ecr.us-east-1.amazonaws.com/graphstorm:local-gpu'


Create a GraphStorm Container
..............................

First, you need to create a GraphStorm container based on the Docker image built in the previous step.

Run the following command:

.. code:: bash

    docker run --gpus all --network=host -v /dev/shm:/dev/shm/ -d --name test graphstorm:local-gpu

Or if using a CPU-only host:

.. code:: bash

    docker run --network=host -v /dev/shm:/dev/shm/ -d --name test graphstorm:local-cpu

This command will create a GraphStorm container, named ``test`` and run the container as a daemon.

Then connect to the container by running the following command:

.. code:: bash

    docker container exec -it test /bin/bash

If successful, the command prompt will change to the container's, like

.. code-block:: console

    root@<ip-address>:/#

.. note::

    If you are preparing the environment to run GraphStorm in a distributed setting, specific instruction for running a Docker image with the NFS folder is given in the :ref:`Use GraphStorm in a Distributed Cluster<distributed-cluster>`.
