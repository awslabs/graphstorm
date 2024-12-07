# GraphStorm SageMaker Pipeline

This project provides a set of tools to create and execute SageMaker pipelines for GraphStorm, a library for large-scale graph neural networks. The pipeline automates the process of graph construction, partitioning, training, and inference using Amazon SageMaker.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Creating a Pipeline](#creating-a-pipeline)
   - [Executing a Pipeline](#executing-a-pipeline)
6. [Pipeline Components](#pipeline-components)
7. [Configuration](#configuration)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

## Overview

This project simplifies the process of running GraphStorm workflows on Amazon SageMaker. It provides scripts to:

1. Define and create SageMaker pipelines for GraphStorm tasks
2. Execute these pipelines with customizable parameters
3. Manage different stages of graph processing, including construction, partitioning, training, and inference

## Prerequisites

- Python 3.8+
- AWS account with appropriate permissions. See the official
  [SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-access.html) docs
  for detailed permissions needed to create and run SageMaker Pipelines.
- Familiarity with SageMaker AI and
  [SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html).
- Basic understanding of graph neural networks and GraphStorm

## Project Structure

The project consists of three main Python scripts:

1. `create_sm_pipeline.py`: Defines the structure of the SageMaker pipeline
2. `pipeline_parameters.py`: Manages the configuration and parameters for the pipeline
3. `execute_pipeline.py`: Executes created pipelines

## Access code and install dependencies

1. Clone the GraphStorm repository:
   ```
   git clone https://github.com/awslabs/graphstorm.git
   cd graphstorm/sagemaker/pipeline
   ```

2. Install the required dependencies:
   ```
   pip install sagemaker boto3
   ```

## Usage

### Creating a Pipeline

To create a new SageMaker pipeline for GraphStorm:

```bash
python create_sm_pipeline.py \
    --role arn:aws:iam::123456789012:role/SageMakerRole \
    --region us-west-2 \
    --graphstorm-pytorch-image-url 123456789012.dkr.ecr.us-west-2.amazonaws.com/graphstorm:sm-cpu \
    --instance-count 2 \
    --jobs-to-run gconstruct train inference \
    --graph-name my-graph \
    --graph-construction-config-filename my_gconstruct_config.json \
    --input-data-s3 s3://input-bucket/data \
    --output-prefix-s3 s3://output-bucket/results \
    --train-inference-task node_classification \
    --train-yaml-s3 s3://config-bucket/train.yaml
```

This command creates a new pipeline with the specified configuration. The pipeline will
include one GConstruct job, one training job and one inference job.
It will use the configuration defined in `s3://input-bucket/data/my_gconstruct_config.json`
to construct the graph and the train config file at `s3://config-bucket/train.yaml`
to run training and inference.

The `--instance-count` parameter determines the number of workers and partitions we will create and use
during partitioning/training. It is also aliased to `--num-parts`.

You can customize various aspects of the pipeline using additional command-line arguments. Refer to the script's help message for a full list of options:

```bash
python create_sm_pipeline.py --help
```

### Executing a Pipeline

To execute a created pipeline:

```bash
python execute_pipeline.py \
    --pipeline-name my-graphstorm-pipeline \
    --region us-west-2
```

You can override various pipeline parameters during execution:

```bash
python execute_pipeline.py \
    --pipeline-name my-graphstorm-pipeline \
    --region us-west-2 \
    --instance-count 4 \
    --gpu-instance-type ml.g4dn.12xlarge
```

For a full list of execution options:

```bash
python execute_pipeline.py --help
```

For more fine-grained execution options see the
[SageMaker AI documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-selective-ex.html).

## Pipeline Components

The GraphStorm SageMaker pipeline typically includes the following steps:

1. **Graph Construction**: Builds the graph from input data
2. **Graph Partitioning**: Partitions the graph for distributed processing
3. **Training**: Trains the graph neural network model
4. **Inference**: Runs inference on the trained model

Each step is configurable and can be customized based on your specific requirements.

## Configuration

The pipeline's behavior is controlled by various configuration parameters, including:

- AWS configuration (region, roles, image URLs)
- Instance configuration (instance types, counts)
- Task configuration (graph name, input/output locations)
- Training and inference configurations

Refer to the `PipelineArgs` class in `pipeline_parameters.py` for a complete list of configurable options.

## Advanced Usage

### Using GraphBolt

To use GraphBolt for improved performance:

```bash
python create_sm_pipeline.py \
    ... \
    --use-graphbolt true
```

### Custom Job Sequences

You can customize the sequence of jobs in the pipeline using the `--jobs-to-run` argument when creating the pipeline. For example:

```bash
python create_sm_pipeline.py \
    ... \
    --jobs-to-run gsprocessing dist_part gb_convert train inference \
    --use-graphbolt true
```

will create a pipeline that uses GSProcessing to process and prepare the data for partitioning,
use GSPartition to partition the data, convert the partitioned data to the GraphBolt format,
then run a train and an inference job in sequence.


### Asynchronous Execution

To start a pipeline execution without waiting for it to complete:

```bash
python execute_pipeline.py \
    --pipeline-name my-graphstorm-pipeline \
    --region us-west-2 \
    --async-execution
```

## Troubleshooting

- Ensure all required AWS permissions are correctly set up
- Check SageMaker execution logs for detailed error messages
- Verify that all S3 paths are correct and accessible. Note trailing `/` that could cause issues.
- Ensure that the specified EC2 instance types are available in your region

See also [Troubleshooting Amazon SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-troubleshooting.html)

For more detailed information about GraphStorm, refer to the [GraphStorm documentation](https://graphstorm.readthedocs.io/).

If you encounter any issues or have questions, please open an issue in the project's [GitHub repository](https://github.com/awslabs/graphstorm/issues).
