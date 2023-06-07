.. _distributed-sagemaker:

Use GraphStorm based on SageMaker
===================================
GraphStorm can run on Amazon Sagemaker to leverage SageMaker's ML DevOps capabilities.

Prerequisites
-----------------
In order to use GraphStorm in Amazon SageMaker, users need to have AWS access to the following AWS services.

- SageMaker service. Please refer to `Anmazon SageMaker service <https://aws.amazon.com/pm/sagemaker/>`_ for how to get access to Amazon SageMaker.
- Amazon ECR. Please refer to `Amazon Elastic Container Registry service <https://aws.amazon.com/ecr/>`_ for how to get access to Amazon ECR.
- S3 service. Please refer to `Amazon S3 service <https://aws.amazon.com/s3/>`_ for how to get access to Amazon S3.
- SageMaker Framework Containers. Please follow `AWS Deep Learning Containers guideline <https://github.com/aws/deep-learning-containers>`_ to get access to the image.
- Amazon EC2 (optional). Please refer to `Amazon EC2 service <https://aws.amazon.com/ec2/>`_ for how to get access to Amazon ECR.

Setup GraphStorm SageMaker Docker Repository
----------------------------------------------
GraphStorm uses SageMaker's "Bring Your Own Container" mode. Therefore, before launch GraphStorm on SageMaker, there are two steps required to set up a GraphStorm SageMaker Docker repository.

Step 1: Build a SageMaker compatible Docker image
...................................................

.. note::
    * Please make sure your account has AWS access key and security access key configured to authenticate access to AWS services.
    * For more details of Amazon ECR operation via CLI, users can refer to `Using Amazon ECR with the AWS CLI document <https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html>`_

First, in either a Linux EC2 instance or a SageMaker Notebook Linux instance, configure a Docker environment by following the `Docker documentent <https://docs.docker.com/get-docker/>`_ suggestions.

In order to use the SageMaker base Docker image, users need to authenticate to use the image with the following command.

.. code-block:: bash

    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

Then, use the following commands to clone GraphStorm source code, and build a GraphStorm SageMaker compatible Docker image from source.

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git
    
    cd /path-to-graphstorm/docker/

    bash /path-to-graphstorm/docker/build_docker_sagemaker.sh /path-to-graphstorm/ docker_name docker_tag

The ``build_docker_sagemaker.sh`` command take three arguments:

1. **path-to-graphstorm** (**required**), is the absolute path of the "graphstorm" folder, where you clone and download the GraphStorm source code. For example, the path could be ``/code/graphstorm``.
2. **docker-name** (optional), is the assigned name of the to be built Docker image. Default is ``graphstorm``.

.. note::
    In order to upload the GraphStorm SageMaker Docker image to ECR, users need to define the <docker-name> to include the ECR URI string, <aws_acount_id>.dkr.ecr.<region>.amazonaws.com, e.g., ``911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm``.

3. **docker-tag** (optional), is the assigned tag name of the to be built docker image. Default is ``sm``.

Once the ``build_docker_sagemaker.sh`` command completes successfully. There will be a Docker image, named ``graphstorm_name:docker_tag``, such as ``<aws_acount_id>.dkr.ecr.<region>.amazonaws.com/graphstorm:sm``, in the local repository.

Step 2: Upload the Docker Image to ECR
........................................
Because SageMaker relies on Amazon ECR to access customers' own Docker images, users need to upload the Docker image built in the Step 1 to their own ECR repository.

The following command will authenticate the user account to access to user's ECR repository via AWS CLI.

.. code-block:: bash

    aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws_acount_id>.dkr.ecr.<region>.amazonaws.com

Please replace the `<region>` and `<aws_acount_id>` with your own information and be consistent with the values used in the Step 1.

In addition, users need to create an ECR repository at the specified `<region>` with the name as `<graphstorm_name>` without the ECR URI string, e.g., ``graphstorm``.

And then use below command to push the built GraphStorm Docker image to users' own ECR repository.

.. code-block:: bash

    docker push <aws_acount_id>.dkr.ecr.<region>.amazonaws.com/<graphstorm_name:docker_tag>

Please replace the `<aws_acount_id>`, `<region>`, and `<graphstorm_name:docker_tag>` with the actual Docker image name, e.g.,  ``911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm:sm``.

Run GraphStorm on SageMaker
----------------------------
There are two ways to run GraphStorm on SageMaker.

* Run with Amazon SageMaker service. This is the formal way to run GraphStorm experiments on large graphs and to deploy GraphStorm on SageMaker for production.
* Run with Docker composes in local environment. This is only for model developers and testers to simulate running GraphStorm on SageMaker.

Install SageMaker
...................

Launch inference
.................
