# Building Docker images for using GSProcessing on SageMaker

We include two Bash scripts to help you build and push a GSProcessing
image to ECR to enable training on Amazon SageMaker.

To build the image you will use `build_gsprocessing_image.sh` and to
push it to ECR you will use `push_gsprocessing_image.sh`.

For a tutorial on building and pushing the images to ECR to use
with Amazon SageMaker see https://graphstorm.readthedocs.io/en/latest/gs-processing/usage/distributed-processing-setup.html.

## Building the image

To build the image you will run `bash build_gsprocessing_image.sh`
script that has one required parameter, `--environment` that
determines the intended execution environment of the image.
We currently support either `sagemaker` or `emr-serverless`.

The script copies the necessary code, optionally builds and packages
the library as a `wheel` file and builds and tags the image.

You can get the other parameters of the script using
`bash build_gsprocessing_image.sh -h/--help` that include:

* `-e, --environment`   Intended execution environment, must be one of `sagemaker` or `emr-serverless`. Required.
* `-p, --path`          Path to graphstorm-processing directory, default is one level above this script.
* `-i, --image`         Docker image name, default is 'graphstorm-processing'.
* `-v, --version`       Docker version tag, default is the library's current version (`poetry version --short`)
* `-b, --build`         Docker build directory, default is `/tmp/`
* `-a, --architecture`  Target architecture for the image. Both execution environments support `x86_64`, while
                        EMR Serverless also supports `arm64`.
* `-s, --suffix`        A suffix to add to the image tag, e.g. `-test` will name the image
                        `graphstorm-processing-${ENVIRONMENT}:${VERSION}-${ARCH}-test`.
* `-t, --target`        Target of the image. Use `test` if you intend to use the image for testing
                        new library functionality, otherwise `prod`. Default: `prod`
* `-m, --hf-model`      When provided with a valid Huggingface model name, will include it in the image. Default is "", no model included.

## Pushing the image

After having built the image you will run `bash push_gsprocessing_image.sh`
to push the image to ECR. By default the script will optionally create
a repository on ECR named `graphstorm-processing-${ENVIRONMENT}` in the `us-west-2` region
and push the image we just built to it.

You can change these default values using the other parameters of the script:

* `-e, --environment`   Intended execution environment, must be one of `sagemaker` or `emr-serverless`. Required.
* `-i, --image`         Docker image name prefix, default is `graphstorm-processing-${ENVIRONMENT}`.
* `-v, --version`       Docker version tag, default is the library's current version (`poetry version --short`)
* `-r, --region`        AWS Region to which we'll push the image. By default will get from aws-cli configuration.
* `-a, --account`       AWS Account ID. By default will get from aws-cli configuration.

## Testing the image

If you build the image with the argument `--target test` the
build script will include the source and tests on the image.

To run the unit tests inside on a container running you have created, which helps ensure the deployed container will
behave as expected, you can run `docker run -it --rm --name gsp  graphstorm-processing-${ENV}:0.2.2-${ARCH}${SUFFIX}`
which will execute the library's unit tests inside a local instance of the provided image.
