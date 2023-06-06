.. _distributed-sagemaker:

Use GraphStorm in SageMaker
============================
GraphStorm can run on Amazon Sagemaker to leverage SageMaker's ML DevOps capabilities.

Prerequisites
-----------------
In order to use GraphStorm in Amazon SageMaker, users need to have AWS access to the following AWS services.

- SageMaker service. Please refer to `Anmazon SageMaker service <https://aws.amazon.com/pm/sagemaker/>`_ for how to get access to Amazon SageMaker.
- Amazon ECR. Please refer to `Amazon Elastic Container Registry service <https://aws.amazon.com/ecr/>`_ for how to get access to Amazon ECR.
- S3 service. Please refer to `Amazon S3 service <https://aws.amazon.com/s3/>`_
- SageMaker Framework Containers. Please follow `AWS Deep Learning Containers guideline <https://github.com/aws/deep-learning-containers>`_ to get access to the image.
- Amazon EC2 (optional). Please refer to `Amazon EC2 service <https://aws.amazon.com/ec2/>`_ for how to get access to Amazon ECR.

Setup SageMaker Environment
------------------------------
Before launch GraphStorm in SageMaker, there are ?? steps required to set up a GraphStorm SageMaker running enviroment.

Step 1: Build a SageMaker compatible Docker image
...................................................


Launch Training
-----------------


Launch inference
----------------
