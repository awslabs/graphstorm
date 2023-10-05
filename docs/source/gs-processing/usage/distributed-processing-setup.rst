GraphStorm Processing setup for Amazon SageMaker
================================================

In this guide we'll demonstrate how to prepare your environment to run
GraphStorm Processing (GSProcessing) jobs on Amazon SageMaker.

We're assuming a Linux host environment used throughout
this tutorial, but other OS should work fine as well.

The steps required are:

- Clone the GraphStorm repository.
- Install Docker.
- Install Poetry.
- Set up AWS access.
- Build the GraphStorm Processing image using Docker.
- Push the image to the Amazon Elastic Container Registry (ECR).
- Launch a SageMaker Processing job using the example scripts.

Clone the GraphStorm repository
-------------------------------

You can clone the GraphStorm repository using

.. code-block:: bash

    git clone https://github.com/awslabs/graphstorm.git

You can then navigate to the ``graphstorm-processing/docker`` directory
that contains the relevant code:

.. code-block:: bash

    cd ./graphstorm/graphstorm-processing/docker

Install Docker
--------------

To get started with building the GraphStorm Processing image
you'll need to have the Docker engine installed.


To install Docker follow the instructions at the
`official site <https://docs.docker.com/engine/install/>`_.

Install Poetry
--------------

We use `Poetry <https://python-poetry.org/docs/>`_ as our build
tool and for dependency management,
so we need to install it to facilitate building the library.

You can install Poetry using:

.. code-block:: bash

    curl -sSL https://install.python-poetry.org | python3 -

For detailed installation instructions the
`Poetry docs <https://python-poetry.org/docs/>`_.


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

Your role should have full ECR access to be able to pull from ECR to build the image,
create an ECR repository if it doesn't exist, and push the GSProcessing image to the repository.

Building the GraphStorm Processing image using Docker
-----------------------------------------------------

Once Docker and Poetry are installed, and your AWS credentials are set up,
we can use the provided scripts
in the ``graphstorm-processing/docker`` directory to build the image.

The ``build_gsprocessing_image.sh`` script can build the image
locally and tag it. For example, assuming our current directory is where
we cloned ``graphstorm/graphstorm-processing``:

.. code-block:: bash

    bash docker/build_gsprocessing_image.sh

The above will use the Dockerfile of the latest available GSProcessing version,
build an image and tag it as ``graphstorm-processing:${VERSION}`` where
``${VERSION}`` will take be the latest available GSProcessing version (e.g. ``0.1.0``).

The script also supports other arguments to customize the image name,
tag and other aspects of the build. See ``bash docker/build_gsprocessing_image.sh --help``
for more information.

Push the image to the Amazon Elastic Container Registry (ECR)
-------------------------------------------------------------

Once the image is built we can use the ``push_gsprocessing_image.sh`` script
that will create an ECR repository if needed and push the image we just built.

The script does not require any arguments and by default will
create a repository named ``graphstorm-processing`` in the ``us-west-2`` region,
on the default AWS account ``aws-cli`` is configured for,
and push the image tagged with the latest version of GSProcessing.

The script supports 4 optional arguments:

1. Image name/repository. (``-i/--image``) Default: ``graphstorm-processing``
2. Image tag. Default: (``-v/--version``) ``<latest_library_version>`` e.g. ``0.1.0``.
3. ECR region. Default: (``-r/--region``) ``us-west-2``.
4. AWS Account ID. (``-a/--account``) Default: Uses the account ID detected by the ``aws-cli``.

Example:

.. code-block:: bash

    bash push_gsprocessing_image.sh -i "graphstorm-processing" -v "0.1.0" -r "us-west-2" -a "1234567890"


Launch a SageMaker Processing job using the example scripts.
------------------------------------------------------------

Once the setup is complete, you can follow the
:doc:`SageMaker Processing job guide <amazon-sagemaker>`
to launch your distributed processing job using AWS resources.
