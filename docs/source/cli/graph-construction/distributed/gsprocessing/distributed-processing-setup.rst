.. _gsprocessing_distributed_setup:

GSProcessing Distributed Setup
=======================================

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
- Launch a SageMaker Processing or EMR Serverless job using the example scripts.

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

GSProcessing supports Amazon SageMaker, EMR on EC2, and EMR Serverless as
execution environments, so we need to choose which image we want
to build first.

The ``build_gsprocessing_image.sh`` script can build the image
locally and tag it, provided the intended execution environment,
using the ``-e/--environment`` argument. The supported environments
are ``sagemaker``, ``emr``, and ``emr-serverless``.
For example, assuming our current directory is where
we cloned ``graphstorm/graphstorm-processing``, we can use
the following to build the SageMaker image:

.. code-block:: bash

    bash docker/build_gsprocessing_image.sh --environment sagemaker

The above will use the SageMaker-specific Dockerfile of the latest available GSProcessing version,
build an image and tag it as ``graphstorm-processing-sagemaker:${VERSION}-x86_64`` where
``${VERSION}`` will take be the latest available GSProcessing version.

The script also supports other arguments to customize the image name,
tag and other aspects of the build. See ``bash docker/build_gsprocessing_image.sh --help``
for more information.

Packaging Huggingface models into the image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you plan to use text transformations (see :ref:`gsp-supported-transformations-ref`)
that utilize Huggingface models, you can opt to include the Huggingface model cache directly in your Docker image.
The ``build_gsprocessing_image.sh`` script provides an option to embed the Huggingface model cache within the Docker image, using the ``--hf-model`` argument.
You can do this for both the SageMaker and EMR Serverless images. It is a good way to save cost as it avoids downloading models after launching the job.
If you'd rather download the Huggingface models at runtime, for EMR Serverless images, setting up a VPC and NAT route is a necessary.
You can find detailed instructions on creating a VPC for EMR Serverless in the AWS documentation: `Create a VPC on emr-serverless
<https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/vpc-access.html>`_.


.. code-block:: bash

    bash docker/build_gsprocessing_image.sh --environment sagemaker --hf-model bert-base-uncased
    bash docker/build_gsprocessing_image.sh --environment emr-serverless --hf-model bert-base-uncased

Support for arm64 architecture
------------------------------

For EMR and EMR Serverless images, it is possible to build images for the ``arm64`` architecture,
which can lead to improved runtime and cost compared to ``x86_64``. For more details
on EMR Serverless architecture options see the
`official docs <https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/architecture.html>`_.

You can build an ``arm64``
image natively by installing Docker and following the above process on an ARM instance such
as ``M6G`` or ``M7G``. See the `AWS documentation <https://aws.amazon.com/ec2/graviton/>`_
for instances powered by the Graviton processor.

To build ``arm64`` images
on an ``x86_64`` host you need to enable multi-platform builds for Docker. The easiest way
to do so is to use QEMU emulation. To install the QEMU related libraries you can run

On Ubuntu

.. code-block:: bash

    sudo apt install -y qemu binfmt-support qemu-user-static

On Amazon Linux/CentOS:

.. code-block:: bash

    sudo yum instal -y qemu-system-arm qemu qemu-user qemu-kvm qemu-kvm-tools \
        libvirt virt-install libvirt-python libguestfs-tools-c

Finally you'd need to ensure ``binfmt_misc`` is configured for different platforms by running

.. code-block:: bash

    docker run --privileged --rm tonistiigi/binfmt --install all

To verify your Docker installation is ready for multi-platform builds you can run:

.. code-block:: bash

    docker buildx ls

    NAME/NODE   DRIVER/ENDPOINT STATUS  BUILDKIT     PLATFORMS
    default *   docker
    default     default         running v0.8+unknown linux/amd64, linux/arm64

Finally, to build an EMR Serverless GSProcessing image for the ``arm64`` architecture you can run:

.. code-block:: bash

    bash docker/build_gsprocessing_image.sh --environment emr-serverless --architecture arm64

.. note::

    Building images for the first time under emulation using QEMU
    can be significantly slower than native builds
    (more than 20 minutes to build the GSProcessing ``arm64`` image).
    After the first build, follow up builds that only change the GSProcessing code
    will be less than a minute thanks to Docker's caching.
    To speed up the build process you can build on an ARM-native instance,
    look into using ``buildx`` with multiple native nodes, or use cross-compilation.
    See `the official Docker documentation <https://docs.docker.com/build/building/multi-platform/>`_
    for details.

Push the image to the Amazon Elastic Container Registry (ECR)
-------------------------------------------------------------

Once the image is built we can use the ``push_gsprocessing_image.sh`` script
that will create an ECR repository if needed and push the image we just built.

The script again requires us to provide the intended execution environment using
the ``-e/--environment`` argument,
and by default will create a repository named ``graphstorm-processing-<environment>`` in the ``us-west-2`` region,
on the default AWS account ``aws-cli`` is configured for,
and push the image tagged with the latest version of GSProcessing.

The script supports 4 optional arguments:

1. Image name/repository. (``-i/--image``) Default: ``graphstorm-processing-<environment>``
2. Image tag. (``-v/--version``) Default: ``<latest_library_version>`` e.g. ``0.2.2``.
3. ECR region. (``-r/--region``) Default: ``us-west-2``.
4. AWS Account ID. (``-a/--account``) Default: Uses the account ID detected by the ``aws-cli``.

Example:

.. code-block:: bash

    bash docker/push_gsprocessing_image.sh -e sagemaker -r "us-west-2" -a "1234567890"

To push an EMR Serverless ``arm64`` image you'd similarly run:

.. code-block:: bash

    bash docker/push_gsprocessing_image.sh -e emr-serverless --architecture arm64 \
        -r "us-west-2" -a "1234567890"

.. _gsp-upload-data-ref:

Upload data to S3
-----------------

For distributed jobs we use S3 as our storage source and target, so before
running any example
we'll need to upload our data to S3. To do so you will need
to have read/write access to an S3 bucket, and the requisite AWS credentials
and permissions.

We will use the AWS CLI to upload data so make sure it is
`installed <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_
and `configured <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html>`_
in you local environment.

Assuming ``graphstorm/graphstorm-processing`` is our current working
directory we can upload the data to S3 using:

.. code-block:: bash

    MY_BUCKET="enter-your-bucket-name-here"
    REGION="bucket-region" # e.g. us-west-2
    aws --region ${REGION} s3 sync ./tests/resources/small_heterogeneous_graph/ \
        "s3://${MY_BUCKET}/gsprocessing-input"

.. note::

    Make sure you are uploading your data to a bucket
    that was created in the same region as the ECR image
    you pushed.

Launch a SageMaker Processing job using the example scripts.
------------------------------------------------------------

Once the setup is complete, you can follow the
:ref:`SageMaker Processing job guide<gsprocessing_distributed_setup>`
to launch your distributed processing job using Amazon SageMaker resources.

Launch an EMR Serverless job using the example scripts.
------------------------------------------------------------

In addition to Amazon SageMaker you can also use EMR Serverless
as an execution environment to allow you to scale to even larger datasets
(recommended when your graph has 30B+ edges).
Its setup is more involved than Amazon SageMaker, so we only recommend
it for experienced AWS users.
Follow the :ref:`EMR Serverless job guide<gsprocessing_emr_serverless>`
to launch your distributed processing job using EMR Serverless resources.
