================================================
Running distributed processing jobs on AWS Infra
================================================

After successfully building the appropriate Docker image and pushing to the Amazon ECR,
you can now initiate GSProcessing jobs utilizing AWS resources.

We support running GSProcessing jobs on different AWS execution environments including: Amazon SageMaker, EMR Serverless, and EMR on EC2.


Running distributed jobs on Amazon SageMaker:

.. toctree::
  :maxdepth: 1
  :titlesonly:

  amazon-sagemaker.rst

Running distributed jobs on EMR Serverless:

.. toctree::
  :maxdepth: 1
  :titlesonly:

  emr-serverless.rst

Running distributed jobs on EMR on EC2:

.. toctree::
  :maxdepth: 1
  :titlesonly:

  emr.rst