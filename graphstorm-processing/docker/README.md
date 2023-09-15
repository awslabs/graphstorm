# Building Docker images for using GSProcessing on SageMaker

We include two Bash scripts to help you build and push a GSProcessing
image to ECR to enable training on Amazon SageMaker.

To build the image you will use `build_gsprocessing_image.sh` and to
push it to ECR you will use `push_gsprocessing_image.sh`.

For a tutorial on building and pushing the images to ECR to use
with Amazon SageMaker see docs/source/usage/distributed-processing-setup.rst.

## Building the image

To build the image you will run `bash build_gsprocessing_image.sh`
script that has one required parameter, `--target` that can take
one of two values, `prod` and `test` that determine whether we
include the source and tests on the image (when `test` is used),
or just install the libary on the image (when `prod` is used).

The script copies the necessary code, optionally builds and packages
the library as a `wheel` file and builds and tags the image.

You can get the other parameters of the script using
`bash build_gsprocessing_image.sh -h/--help` that include:

* `-p, --path`      Path to graphstorm-processing directory, default is one level above this script.
* `-i, --image`     Docker image name, default is 'graphstorm-processing'.
* `-v, --version`   Docker version tag, default is the library's current version (`poetry version --short`)
* `-b, --build`     Docker build directory, default is '/tmp/`



## Pushing the image

After having built the image you will run `bash push_gsprocessing_image.sh`
to push the image to ECR. By default the script will optionally create
a repository on ECR named `graphstorm-processing` in the `us-west-2` region
and push the image we just built to it.

You can change these default values using the other parameters of the script:

* `-i, --image`     Docker image name, default is 'graphstorm-processing'.
* `-v, --version`   Docker version tag, default is the library's current version (`poetry version --short`)
* `-r, --region`    AWS Region to which we'll push the image. By default will get from aws-cli configuration.
* `-a, --account`   AWS Account ID. By default will get from aws-cli configuration.