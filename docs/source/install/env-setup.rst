.. _setup:

Environment Setup
======================
GraphStorm supports two environment setup methods:
    - Install GraphStorm using as a pip packcage. This method works well for development and test on a signle machine.
    - Setup a GraphStorm Docker image. This method is good for using GraphStorm in distributed environments that commonly used in production.

.. _setup_pip:

1. Setup GraphStorm with pip Packages
--------------------------------------
Prerequisites
...............

1. **Linux OS**: The current version of GraphStorm supports Linux as the Operation System. We tested GraphStorm on both Ubuntu (22.04 or later version) and Amazon Linux 2.

2. **Python3**: The current version of GraphStorm requires Python installed with the version larger than **3.7**.

3. (Optional) GraphStorm supports **Nvidia GPUs** for using GraphStorm in GPU environments.

Install GraphStorm
...................
Users can use ``pip`` or ``pip3`` to install GraphStorm.

.. code-block:: bash

    pip install graphstorm

Install Dependencies
.....................
Users should install PyTorch v1.13.1 and DGL v1.0.4 that is the core dependency of GraphStorm using the following commands.

For Nvidia GPU environment:

.. code-block:: bash

    pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install dgl==1.0.4+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html

For CPU environment:

.. code-block:: bash

    pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
    pip install dgl==1.0.4 -f https://data.dgl.ai/wheels-internal/repo.html

Configure SSH No-password login
................................
Use the following commands to configure a local SSH no-password login that GraphStorm relies on.

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

.. warning:: 

    - If use this method to setup GraphStorm environment, please replace the argument ``--ssh-port`` of in launch commands in GraphStorm's tutorials from 2222 with **22**.

    - If use this method to setup GraphStorm environment, you may need to replace the ``python3`` command with ``python``, depending on your Python versions.

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

Please use the following command to build a Docker image from source:

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git

    cd /path-to-graphstorm/docker/

    bash /path-to-graphstorm/docker/build_docker_oss4local.sh /path-to-graphstorm/ docker-name docker-tag

There are three arguments of the ``build_docker_oss4local.sh``:

1. **path-to-graphstorm** (**required**), is the absolute path of the "graphstorm" folder, where you cloned the GraphStorm source code. For example, the path could be ``/code/graphstorm``.
2. **docker-name** (optional), is the assigned name of the to be built Docker image. Default is ``graphstorm``.
3. **docker-tag** (optional), is the assigned tag name of the to be built docker image. Default is ``local``.

You can use the below command to check if the new Docker image is created successfully.

.. code:: bash

    docker image ls

If the build succeeds, there should be a new Docker image, named *<docker-name>:<docker-tag>*, e.g., ``graphstorm:local``.

Create a GraphStorm Container
..............................

First, you need to create a GraphStorm container based on the Docker image built in the previous step.

Run the following command:

.. code:: bash

    nvidia-docker run --network=host -v /dev/shm:/dev/shm/ -d --name test graphstorm:local

This command will create a GraphStorm container, named ``test`` and run the container as a daemon.

Then connect to the container by running the following command:

.. code:: bash

    docker container exec -it test /bin/bash

If succeeds, the command prompt will change to the container's, like

.. code-block:: console

    root@<ip-address>:/#
