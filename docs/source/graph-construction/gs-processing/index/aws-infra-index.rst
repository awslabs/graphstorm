================================================
Running distributed processing jobs on AWS Infra
================================================

After successfully building the appropriate Docker image and pushing it to
`Amazon ECR <https://docs.aws.amazon.com/ecr/>`_,
you can now initiate GSProcessing jobs utilizing AWS resources.

We support running GSProcessing jobs on different AWS execution environments including:
`Amazon SageMaker <https://docs.aws.amazon.com/sagemaker/>`_,
`EMR Serverless <https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/emr-serverless.html>`_, and
`EMR on EC2 <https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-what-is-emr.html>`_.


Running distributed jobs on `Amazon SageMaker <https://docs.aws.amazon.com/sagemaker/>`_:

.. toctree::
  :maxdepth: 1
  :titlesonly:

  ../usage/amazon-sagemaker.rst

Running distributed jobs on `EMR Serverless <https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/emr-serverless.html>`_:

.. toctree::
  :maxdepth: 1
  :titlesonly:

  ../usage/emr-serverless.rst

Running distributed jobs on `EMR on EC2 <https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-what-is-emr.html>`_:

.. toctree::
  :maxdepth: 1
  :titlesonly:

  ../usage/emr.rst