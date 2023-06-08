.. _setup:

Environment Setup
======================

For a quick and easy setup, GraphStorm can be installed as a pip package.

However, configuring an GraphStorm environment is non-trivial. Users need to install dependencies and configure distributed PyTorch running environments. For this reason, we recommend that our users setup Docker as the base running environment to use GraphStorm.

Prerequisites
-----------------

1. **Docker**: You need to install Docker in your environment as the `Docker documentation <https://docs.docker.com/get-docker/>`_ suggests, and the `Nvidia Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_.

For example, in an AWS EC2 instance without Docker preinstalled, you can run the following commands to install Docker.

.. code-block:: bash

    sudo apt-get update
    sudo apt update
    sudo apt install Docker.io

If using AWS `Deep Learning AMI GPU version`, the Nvidia Container Toolkit has been preinstalled.

2. **GPU**: The current version of GraphStorm requires **at least one GPU** installed in the instance.

.. _build_docker:

Build a GraphStorm Docker image from source code
--------------------------------------------------

Please use the following command to build a Docker image from source:

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git
    
    cd /path-to-graphstorm/docker/

    bash /path-to-graphstorm/docker/build_docker_oss4local.sh /path-to-graphstorm/ docker-name docker-tag

There are three arguments of the ``build_docker_oss4local.sh``:

1. **path-to-graphstorm** (**required**), is the absolute path of the "graphstorm" folder, where you clone and download the GraphStorm source code. For example, the path could be ``/code/graphstorm``.
2. **docker-name** (optional), is the assigned name of the to be built Docker image. Default is ``graphstorm``.
3. **docker-tag** (optional), is the assigned tag name of the to be built docker image. Default is ``local``.

You can use the below command to check if the new Docker image is created successfully.

.. code:: bash

    docker image ls

If the build succeeds, there should be a new Docker image, named *<docker-name>:<docker-tag>*, e.g., ``graphstorm:local``.


Create a GraphStorm Container
-------------------------------

First, you need to create a GraphStorm container based on the Docker image built in the previous step. 

Run the following command:

.. code:: bash

    nvidia-docker run --network=host -v /dev/shm:/dev/shm/ -d --name test graphstomr:local

This command will create a GraphStorm contained, named ``test`` and run the container as a daemon. 

Then connect to the container by running the following command:

.. code:: bash

    docker container exec -it test /bin/bash

If succeeds, the command prompt will change to the container's, like

.. code-block:: console

    root@ip-address:/#
