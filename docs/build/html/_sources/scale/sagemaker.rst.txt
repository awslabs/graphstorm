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
GraphStorm uses SageMaker's "Bring Your Own Container (BYOC)" mode. Therefore, before launch GraphStorm on SageMaker, there are two steps required to set up a GraphStorm SageMaker Docker repository.

.. _build_docker:

Step 1: Build a SageMaker compatible Docker image
...................................................

.. note::
    * Please make sure your account has AWS access key (AK) and security access key (SK) configured to authenticate access to AWS services.
    * For more details of Amazon ECR operation via CLI, users can refer to `Using Amazon ECR with the AWS CLI document <https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html>`_.

First, in either a Linux EC2 instance or a SageMaker Notebook Linux instance, configure a Docker environment by following the `Docker documentent <https://docs.docker.com/get-docker/>`_ suggestions.

In order to use the SageMaker base Docker image, users need to authenticate to use SageMaker images with the following command.

.. code-block:: bash

    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

Then, clone GraphStorm source code, and build a GraphStorm SageMaker compatible Docker image from source with commands:

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git
    
    cd /path-to-graphstorm/docker/

    bash /path-to-graphstorm/docker/build_docker_sagemaker.sh /path-to-graphstorm/ <DOCKER_NAME> <DOCKER_TAG>

The ``build_docker_sagemaker.sh`` command take three arguments:

1. **path-to-graphstorm** (**required**), is the absolute path of the "graphstorm" folder, where you clone and download the GraphStorm source code. For example, the path could be ``/code/graphstorm``.
2. **DOCKER_NAME** (optional), is the assigned name of the to be built Docker image. Default is ``graphstorm``.

.. note::
    In order to upload the GraphStorm SageMaker Docker image to ECR, users need to define the <DOCKER_NAME> to include the ECR URI string, <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com, e.g., ``911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm``.

3. **DOCKER_TAG** (optional), is the assigned tag name of the to be built docker image. Default is ``sm``.

Once the ``build_docker_sagemaker.sh`` command completes successfully. There will be a Docker image, named ``<DOCKER_NAME>:<DOCKER_TAG>``, such as ``911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm:sm``, in the local repository.

.. _upload_docker:

Step 2: Upload the Docker Image to ECR
........................................
Because SageMaker relies on Amazon ECR to access customers' own Docker images, users need to upload the Docker image built in the Step 1 to their own ECR repository.

The following command will authenticate the user account to access to user's ECR repository via AWS CLI.

.. code-block:: bash

    aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com

Please replace the `<REGION>` and `<AWS_ACCOUNT_ID>` with your own account information and be consistent with the values used in the Step 1.

In addition, users need to create an ECR repository at the specified `<REGION>` with the name as `<DOCKER_NAME>` **WITHOUT** the ECR URI string, e.g., ``graphstorm``.

And then use below command to push the built GraphStorm Docker image to users' own ECR repository.

.. code-block:: bash

    docker push <DOCKER_NAME>:<DOCKER_TAG>

Please replace the `<DOCKER_NAME>` and `<DOCKER_TAG>` with the actual Docker image name, e.g., ``911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm:sm``.

Run GraphStorm on SageMaker
----------------------------
There are two ways to run GraphStorm on SageMaker.

* Run with Amazon SageMaker service. In this way, users will use GraphStorm's tools to submit SageMaker API calls, which will request SageMaker services to start new SageMaker training or inference instances that run GraphStorm code. Users can submit the API calls in a cheap EC2 instance or a SageMaker Notebook instance without GPUs (e.g., C5.xlarge). This is the formal way to run GraphStorm experiments on large graphs and to deploy GraphStorm on SageMaker for production.
* Run with Docker compose in local environment. In this way, users do not call the SageMaker service, but use Docker compose to run SageMaker locally in an EC2 instance or a SageMaker Notebook instance that has GPUs. This is mainly for model developers and testers to simulate running GraphStorm on SageMaker.

Run GraphStorm with Amazon SageMaker service
..............................................
To call Amazon SageMaker service, users should set up an instance with SageMaker library installed and GraphStorm's SageMaker tools copied.

1. Use the below command to install SageMaker.

.. code-block:: bash

    pip install sagemaker

2. Copy GraphStorm SageMaker tools. Users can clone the GraphStorm repository with the following command, or copy the `sagemaker folder <https://github.com/awslabs/graphstorm/tree/main/sagemaker>`_ to the instance.

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git

Prepare graph data
`````````````````````
Unlike GraphStorm's :ref:`Standalone mode<quick-start-standalone>` and :ref:`the Distributed mode<distributed-cluster>` that rely on local disk or shared file system to store the partitioned graph, SageMaker uses Amaonz S3 as the shared data storage to distribute partitioned graphs and the configuration YAML file.

This tutorial uses the same three-partition OGB-MAG graph and the link prediction task as those introduced in the :ref:`Partition a Graph<partition-a-graph>` section of the :ref:`Use GraphStorm in a Distributed Cluster<distributed-cluster>` tutorial. After generate the partitioned OGB-MAG graphs, use the following commands to upload them and the GraphStorm configuration YAML file to an S3 bucket.

.. code-block:: bash

    aws s3 cp --recursive /data/ogbn_mag_lp_3p s3://<PATH_TO_DATA>/ogbn_mag_lp_3p
    aws s3 cp /graphstorm/training_scripts/gsgnn_lp/mag_lp.yaml s3://<PATH_TO_TRAINING_CONFIG>/mag_lp.yaml

Please replace the `<PATH_TO_DATA>` and `<PATH_TO_TRAINING_CONFIG>` with your own S3 bucket URI.

Launch training 
```````````````````
Launch GraphStorm training on SageMaker is similar as launch in the :ref:`Standalone mode<quick-start-standalone>` and :ref:`the Distributed mode<distributed-cluster>`, except for three diffences:
* The launch command is under the ``graphstorm/sagemaker`` folder, and
* Users need to provide AWS service-related information in the command.
* All paths for saving models, embeddings, and predict results should be an S3 location specified through the ``--model-artifact-s3`` argument.

.. note::
    Before running SageMaker tasks, login to the ECR where the image is present.
    .. code-block:: bash

        aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com

    Please replace the `<REGION>` and `<AWS_ACCOUNT_ID>` with your own account information and be consistent with the values used in the Step 1.

Users can use the following commands to launch a GraphStorm link prediction training job with the OGB-MAG graph.

.. code-block:: bash

    cd /path-to-graphstorm/sagemaker/
    
    python3 launch/launch_train.py \
            --image-url <AMAZON_ECR_IMAGE_URI> \
            --region <REGION> \
            --entry-point run/train_entry.py \
            --role <ROLE_ARN> \
            --instance-count 3 \
            --graph-data-s3 s3://<PATH_TO_DATA>/ogbn_mag_lp_3p \
            --yaml-s3 s3://<PATH_TO_TRAINING_CONFIG>/mag_lp.yaml \
            --model-artifact-s3 s3://<PATH_TO_SAVE_TRAINED_MODEL>/ \
            --graph-name ogbn-mag \
            --task-type link_prediction \
            --lp-decoder-type dot_product \
            --num-layers 1 \
            --fanout 10 \
            --hidden-size 128 \
            --backend gloo \
            --batch-size 128

Please replace the `<AMAZON_ECR_IMAGE_URI>` with the `<DOCKER_NAME>:<DOCKER_TAG>` that used in the Step 2, e.g., ``911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm-sagemaker-oss:v0.1``, replace the `<REGION>` with the region where ECR image repository is located, e.g., ``us-east-1``, and replace the `<ROLE_ARN>` with your AWS account ARN that has SageMaker execution role, e.g., ``"arn:aws:iam::<ACCOUNT_ID>:role/service-role/AmazonSageMaker-ExecutionRole-20220627T143571"``.

Because we use three-partition OGB-MAG graph, we need to set the ``--instance-count`` to 3 in this command.

The trained model artifact will be stored in the S3 address provided through ``--model-artifact-s3``. You can use following command to check the model artifacts:

.. code-block:: bash

    aws s3 ls s3://<PATH_TO_SAVE_TRAINED_MODEL>/

.. note:: the ``save_embed_path`` and ``save_prediction_path`` **MUST** be disabled, i.e., set to 'None' when using SageMaker. They only work with local disk (in the Standalone mode) or shared file system (in the Distributed mode).

Launch inference
`````````````````````
Users can use the following command to launch a GraphStorm link prediction training job with the OGB-MAG graph.

.. code-block:: bash

    python3 launch/launch_infer.py \
            --image-url <AMAZON_ECR_IMAGE_URI> \
            --region <REGION> \
            --entry-point run/infer_entry.py \
            --role <ROLE_ARN> \
            --instance-count 3 \
            --graph-data-s3 s3://<PATH_TO_DATA>/ogbn_mag_lp_3p \
            --yaml-s3 s3://<PATH_TO_TRAINING_CONFIG>/mag_lp.yaml \
            --model-artifact-s3 s3://<PATH_TO_SAVE_TRAINED_MODEL>/ \
            --output-emb-s3 s3://<PATH_TO_SAVE_GENERATED_NODE_EMBEDDING>/ \
            --output-prediction-s3 s3://<PATH_TO_SAVE_PREDICTION_RESULTS> \
            --graph-name ogbn-mag \
            --task-type link_prediction \
            --num-layers 1 \
            --fanout 10 \
            --hidden-size 128 \
            --backend gloo \
            --batch-size 128

The generated node embeddings will be uploaded into ``s3://<PATH_TO_SAVE_GENERATED_NODE_EMBEDDING>/``. For node classification/regression or edge classification/regression tasks, users can use ``--output-prediction-s3`` to specify location of saving prediction results. 

Users can use following command to check the corresponding outputs:

.. code-block:: bash

    aws s3 ls s3://<PATH_TO_SAVE_GENERATED_NODE_EMBEDDING>/
    aws s3 ls s3://<PATH_TO_SAVE_PREDICTION_RESULTS>/

Run GraphStorm SageMaker with Docker Compose
..............................................
This section describes how to launch Docker compose jobs that emulate a SageMaker training execution environment. This can be used to develop and test GraphStorm model training and inference using SageMaker.

If users have never worked with Docker compose before the official description provides a great intro:

.. hint::
    
    Compose is a tool for defining and running multi-container Docker applications. With Compose, you use a YAML file to configure your application's services. Then, with a single command, you create and start all the services from your configuration.

We will use this capability to launch multiple worker instances locally, that will be configured to “look like” a SageMaker training instance and communicate over a virtual network created by Docker compose. This way our test environment will be as close to a real SageMaker distributed job as we can get, without needing to launch SageMaker jobs, or launch and configure multiple EC2 instances when developing features.

Get Started
`````````````
To run GraphStorm SageMaker with Docker compose, we need to set up a local Linux instance with the following contents.

1. Use the below command to install SageMaker.

.. code-block:: bash

    pip install sagemaker

2. Clone GraphStorm and install dependencies.

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git

    pip install boto3==1.26.126
    pip install botocore==1.29.126
    pip install h5py==3.8.0
    pip install scipy
    pip install tqdm==4.65.0
    pip install pyarrow==12.0.0
    pip install transformers==4.28.1
    pip install pandas
    pip install scikit-learn
    pip install ogb==1.3.6
    pip install psutil==5.9.5
    pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
    pip3 install dgl==1.0.0 -f https://data.dgl.ai/wheels/cu116/repo.html

    export PYTHONPATH=/PATH_TO_GRAPHSTORM/python:$PYTHONPATH

3. Build a SageMaker compatible Docker image following the :ref:`Step 1 <build_docker>`.

4. Install `docker compose <https://docs.docker.com/compose/install/linux/>`_.

Generate a Docker Compose file
`````````````````````````````````
A Docker Compose file is a YAML file that tells Docker which containers to spin up and how to configure them. To launch the services with a Docker Compose file, we can use ``docker compose -f docker-compose.yaml up``. This will launch the container and execute its entry point.

To emulate a SageMaker distributed execution environment based on the image (suppose the docker image is named ``graphstorm:sm``) built previously, you would need a Docker Compose file that looks like this:

.. code-block:: yaml

    version: '3.7'

    networks:
    gfs:
        name: gsf-network

    services:
    algo-1:
        image: graphstorm:sm
        container_name: algo-1
        hostname: algo-1
        networks:
        - gsf
        command: 'xxx'
        environment:
        SM_TRAINING_ENV: '{"hosts": ["algo-1", "algo-2", "algo-3", "algo-4"], "current_host": "algo-1"}'
        WORLD_SIZE: 4
        MASTER_ADDR: 'algo-1'
        AWS_REGION: 'us-west-2'
        ports:
        - 22
        working_dir: '/opt/ml/code/'

    algo-2:
        [...]

Some explanation on the above elements (see the `official docs <https://docs.docker.com/compose/compose-file/>`_ for more details):

* **image**: Determines which image you will use for the container launched.
* **environment**: Determines the environment variables that will be set for the container once it launches.
* **command**: Determines the entrypoint, i.e. the command that will be executed once the container launches.

To help users generate yaml file automatically, we provide a Python script, ``generate_sagemaker_docker_compose.py``, that builds the docker compose file for users. 

.. Note:: The script uses the `PyYAML <https://pypi.org/project/PyYAML/>`_ library. Please use the below commnd to install it.

    .. code-block:: bash

        pip install pyyaml

This file has 4 required arguments that determine the Docker Compose file that will be generated:

* **--aws-access-key-id**: The AWS access key ID for accessing S3 data within docker
* **--aws-secret-access-key**: The AWS secret access key for accessing S3 data within docker.
* **--aws-session-token**: The AWS session toekn used for accessing S3 data within docker.
* **--num-instances**: The number of instances we want to launch. This will determine the number of algo-x services entries our compose file ends up with.

The rest of the arguments are passed on to ``sagemaker_train.py`` or ``sagemaker_infer.py``:

* **--task-type**: Task type.
* **--graph-data-s3**: S3 location of the input graph.
* **--graph-name**: Name of the input graph.
* **--yaml-s3**: S3 location of yaml file for training and inference.
* **--custom-script**: Custom training script provided by a customer to run customer training logic. This should be a path to the python script within the docker image.
* **--output-emb-s3**: S3 location to store GraphStorm generated node embeddings. This is an inference only argument.
* **--output-prediction-s3**: S3 location to store prediction results. This is an inference only argument.

Run GraphStorm on Docek Compose for Training
```````````````````````````````````````````````
First, use the following command to generate a Compose YAML file for the Link Prediction training on OGB-MAG graph.

.. code-block:: bash

    python3 generate_sagemaker_docker_compose.py \
            --aws-access-key <<AWS_ACCESS_KEY>> \
            --aws-secret-access-key <AWS_SECRET_ACCESS_KEY> \
            --aws-session-token <AWS_SESSION_TOKEN> \
            --num-instances 3 \
            --image <GRAPHSTORM_DOCKER_IMAGE> \
            --graph-data-s3 s3://<PATH_TO_DATA>/ogbn_mag_lp_3p \
            --yaml-s3 s3://<PATH_TO_TRAINING_CONFIG>/map_lp.yaml \
            --model-artifact-s3 s3://<PATH_TO_SAVE_TRAINED_MODEL> \
            --graph-name ogbn-mag \
            --task-type link_prediction \
            --num-layers 1 \
            --fanout 10 \
            --hidden-size 128 \
            --backend gloo \
            --batch-size 128

The above command will create a Docker compose file named ``docker-compose-<task-type>-<num-instances>-train.yaml``, which we can then use to launch the job. 

As our Docker Compose will use a Docker network, ``gsf-network``, for container communications, users need to run the following command to create the network first.

.. code-block:: bash

    docker network create "gsf-network"

Then, use the following command to run the Link Prediction training on OGB-MAG graph.

.. code-block:: bash

    docker compose -f docker-compose-link_prediction-3-train.yaml up

Running the above command will launch 3 instances of the image, configured with the command and env vars that emulate a SageMaker execution environment and run the sagemaker_train.py script. 

.. Note:: The containers actually interact with S3 so you would require valid AWS credentials to run.

Run GraphStorm on Docek Compose for Inference
```````````````````````````````````````````````
Similar to training, the ``generate_sagemaker_docker_compose.py`` can build Compose file for infernece task with the same arguments as for training, and in addition, adding a new argument, ``--inference``. The below command create the Compose file for the Linke Prediction inference on OGB-MAG graph.

.. code-block:: bash

    python3 generate_sagemaker_docker_compose.py \
            --aws-access-key <<AWS_ACCESS_KEY>> \
            --aws-secret-access-key <AWS_SECRET_ACCESS_KEY> \
            --aws-session-token <AWS_SESSION_TOKEN> \
            --num-instances 3 \
            --image <GRAPHSTORM_DOCKER_IMAGE> \
            --graph-data-s3 s3://<PATH_TO_DATA>/ogbn_mag_lp_3p \
            --yaml-s3 s3://<PATH_TO_TRAINING_CONFIG>/map_lp.yaml \
            --model-artifact-s3 s3://<PATH_TO_SAVE_TRAINED_MODEL> \
            --graph-name ogbn-mag \
            --task-type link_prediction \
            --num-layers 1 \
            --fanout 10 \
            --hidden-size 128 \
            --backend gloo \
            --batch-size 128 \
            --inference

The command will create a Docker compose file named ``docker-compose-<task-type>-<num-instances>-infer.yaml``. And then, we can use the same command to spin up the inference job.

.. code-block:: bash

    docker compose -f docker-compose-link_prediction-3-infer.yaml up
