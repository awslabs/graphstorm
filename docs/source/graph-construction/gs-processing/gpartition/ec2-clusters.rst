======================================
Running partition jobs on EC2 Clusters
======================================

Once the :ref:`distributed processing setup<gsprocessing_distributed_setup>` is completed,
users can start the partition jobs. This doc will provide instructions on how to setup an EC2 cluster and
start GPartition jobs on an EC2 cluster.

Create a GraphStorm Cluster
----------------------------

Setup instances of a cluster
................................
A cluster contains several instances, each of which runs a GraphStorm Docker container.

To create such a cluster please follow the :ref:`Environment Setup <setup_docker>`. The guide shows you how to build GraphStorm Docker images, and use a Docker container registry, e.g. AWS ECR, to upload the GraphStorm image to an ECR repository and pull it on the instances in the cluster.

If you can't access a Docker registry from your environment, in **each** instance of the cluster, follow the :ref:`Environment Setup <setup_docker>` description to build a GraphStorm Docker image, and start the image as a container. Then exchange the ssh key from inside of one GraphStorm Docker containers to the rest containers in the cluster, i.e., copy the keys from the ``/root/.ssh/id_rsa.pub`` from one container to ``/root/.ssh/authorized_keys`` in containers on all other containers.

.. note::

    If you are planning to use **parmetis** algorithm, please prepare your docker container using instructions here:

    .. code-block:: bash

        git clone https://github.com/awslabs/graphstorm.git

        cd /path-to-graphstorm/docker/

        bash /path-to-graphstorm/docker/build_docker_parmetis.sh /path-to-graphstorm/ image-name image-tag

    There are four positional arguments for ``build_docker_parmetis.sh``:

    1. **path-to-graphstorm** (**required**), is the absolute path of the "graphstorm" folder, where you cloned the GraphStorm source code. For example, the path could be ``/code/graphstorm``.
    2. **image-name** (optional), is the assigned name of the to be built Docker image. Default is ``graphstorm``.
    3. **image-tag** (optional), is the assigned tag prefix of the Docker image. Default is ``local``.

    The other stuff should remain the same.

Setup a shared file system for the cluster
...............................................
A cluster requires a shared file system, such as NFS or EFS, mounted to each instance in the cluster, in which all GraphStorm containers in the cluster can share data files, and save model artifacts and prediction results.

`Here <https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage/dist#step-0-setup-a-distributed-file-system>`_ is the instruction of setting up NFS for a cluster provided by DGL. As the steps of setup of an NFS could be various for different systems, we suggest users to look for additional information about NFS setting. Here are some available resources: `NFS tutorial <https://www.digitalocean.com/community/tutorials/how-to-set-up-an-nfs-mount-on-ubuntu-22-04>`_ by DigitalOcean, `NFS document <https://ubuntu.com/server/docs/service-nfs>`_ for Ubuntu, `NFS guide <https://www.linode.com/docs/guides/using-an-nfs-server-on-ubuntu2004/>`_ by Linode, `NFS tutorial <https://www.tecmint.com/how-to-setup-nfs-server-in-linux/>`_ at Tecmint, and `NFS guide <https://www.howtoforge.com/how-to-install-nfs-server-and-client-on-ubuntu-22-04/>`_ by HowtoForge.

For an AWS EC2 cluster, users can also use EFS as the shared file system. Please follow 1) `the instruction of creating EFS <https://docs.aws.amazon.com/efs/latest/ug/gs-step-two-create-efs-resources.html>`_; 2) `the instruction of installing an EFS client <https://docs.aws.amazon.com/efs/latest/ug/installing-amazon-efs-utils.html>`_; and 3) `the instructions of mounting the EFS filesystem <https://docs.aws.amazon.com/efs/latest/ug/efs-mount-helper.html>`_ to set up EFS.

After setting up a shared file system, we can keep all partitioned graph data in a shared folder. Then mount the data folder to the ``/path_to_data/`` of each instances in the cluster so that all GraphStorm containers in the cluster can access these partitioned graph data.

Create GraphStorm container by mounting the NFS folder
.......................................................
In each instance, use the following command to start a GraphStorm Docker container and run it as a backend daemon on cpu.

.. code-block:: shell

    docker run -v /path_to_data/:/data \
                      -v /dev/shm:/dev/shm \
                      --network=host \
                      -d --name test graphstorm:local-cpu service ssh restart

This command mount the shared ``/path_to_data/`` folder to each container's ``/data/`` folder by which GraphStorm codes can access graph data and save training and inference outcomes.

Setup the IP address file and check port status
----------------------------------------------------------

Collect the IP list
......................
The GraphStorm Docker containers use SSH on port ``2222`` to communicate with each other. Users need to collect all IP addresses of all the instances and put them into a text file, e.g., ``/data/ip_list.txt``, which is like:

.. figure:: ../../../../../tutorial/distributed_ips.png
    :align: center

.. note:: If possible, use **private IP addresses**, insteand of public IP addresses. Public IP addresses may have additional port constraints, which cause communication issues.

Put this file into container's ``/data/`` folder.

Check port
................
The GraphStorm Docker container uses port ``2222`` to **ssh** to containers running on other machines without passwords. Please make sure all host instances do not use this port.

Users also need to make sure the port ``2222`` is open for **ssh** commands.

Pick one instance and run the following command to connect to the GraphStorm Docker container.

.. code-block:: bash

    docker container exec -it test /bin/bash

In the container environment, users can check the connectivity with the command ``ssh <ip-in-the-cluster> -o StrictHostKeyChecking=no -p 2222``. Please replace the ``<ip-in-the-cluster>`` with the real IP address from the ``ip_list.txt`` file above, e.g.,

.. code-block:: bash

    ssh 172.38.12.143 -o StrictHostKeyChecking=no -p 2222

If successful, you should login to the container in the ``<ip-in-the-cluster>`` instance.

If not, please make sure there is no restriction of exposing port 2222.

For distributed training, users also need to make sure ports under 65536 is open for DistDGL to use.


Launch GPartition Job
----------------------

Now we can ssh into the **leader node** of the EC2 cluster, and start GPartition process with the following command:

.. code:: bash

    python3 -m graphstorm.gpartition.dist_partition_graph
        --input-path ${LOCAL_INPUT_DATAPATH} \
        --metadata-filename ${METADATA_FILE} \
        --output-path ${LOCAL_OUTPUT_DATAPATH} \
        --num-parts ${NUM_PARTITIONS} \
        --partition-algorithm ${ALGORITHM} \
        --ip-config ${IP_CONFIG} \
        --do-dispatch

.. warning::
    1. Please make sure the both ``LOCAL_INPUT_DATAPATH`` and ``LOCAL_OUTPUT_DATAPATH`` are located on the shared filesystem.
    2. The num of instances in the cluster should be a multiple of ``NUM_PARTITIONS`` here.

Currently we support both ``random`` and ``parmetis`` as the partitioning algorithm for EC2 clusters.
