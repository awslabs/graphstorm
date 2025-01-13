# Faster distributed graph neural network training with GraphStorm 0.4

GraphStorm is a low-code enterprise graph machine learning (ML) framework that provides ML practitioners a simple way of building, training and deploying graph ML solutions on industry-scale graph data. While GraphStorm can run efficiently on single instances for small graphs, it truly shines when scaling to enterprise-level graphs in distributed mode using a cluster of EC2 instances or Amazon SageMaker.

GraphStorm 0.4 introduced integration with DGL-GraphBolt, a new graph storage and sampling framework that uses a compact graph representation and pipelined sampling to reduce memory requirements and speed up Graph Neural Network (GNN) training. In this example we'll show how GraphStorm 0.4 brings inference speedups of up to 4x, and per-epoch training speedup up to 2x on the papers100M dataset, with even larger speedups possible [1].

In this example, you will:

1. Learn how to use SageMaker Pipelines with GraphStorm.
2. Understand how GraphBolt enhances GraphStorm's performance in distributed settings.
3. Follow a hands-on example of using GraphStorm with GraphBolt on Amazon SageMaker for distributed training.

## Background: challenges of graph training

Before diving into our hands-on example, it's important to understand some challenges associated with graph training, especially as graphs grow in size and complexity:

1. Memory Constraints: As graphs grow larger, they may no longer fit into the memory of a single machine. A graph with 1B nodes with 512 features per node and 10B edges will require more than 4TB of memory to store, even with optimal representation.  This necessitates distributed processing and more efficient graph representation.
2. Graph Sampling: In GNN mini-batch training, you need to sample neighbors for each node to propagate their representations. For multi-layer GNNs, this can lead to exponential growth in the number of nodes sampled. Efficient sampling methods become necessary.
3. Remote Data Access: When training on multiple machines, retrieving node features and sampling neighborhoods from other machines will significantly impact performance due to network latency. For example, reading a 1024-feature vector from main memory will take around 3μs, while reading that vector from a remote key/value store would take 50-100x longer.

GraphStorm and GraphBolt help address these challenges through efficient graph representations, smart sampling techniques, and sophisticated partitioning algorithms like ParMETIS.


## GraphBolt: pipeline-driven graph sampling

GraphBolt is a new data loading and graph sampling framework developed by the [DGL](https://www.dgl.ai/) team. It streamlines the operations needed to sample efficiently from a heterogeneous graph and fetch the corresponding features.

GraphBolt introduces a new, more compact graph structure representation for heterogeneous graphs, called fused Compressed Sparse Column (fCSC). This can reduce the memory cost of storing a heterogeneous graph by up to 56%, allowing users to fit larger graphs in memory and potentially use smaller, more cost-efficient instances for GNN model training.


### Integration with GraphStorm:

GraphStorm 0.4.0 seamlessly integrates with GraphBolt, allowing users to leverage these performance improvements in their GNN workflows. This integration enables GraphStorm to handle larger graphs more efficiently and accelerate both training and inference processes.

The integration of GraphBolt into GraphStorm's workflow means that users can now:

1. Train models on larger graphs with fewer hardware resources.
2. Achieve faster training and inference times with more efficient graph sampling framework.
3. Utilize GPU resources more effectively for graph learning.

### Performance improvements:

Our benchmarks show significant improvements in both memory usage and training speed when using GraphStorm with GraphBolt:


* We've observed up to 1.8x training speedup on the [ogbn-papers 100M dataset](https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M), with 111M nodes and 3.2B edges
* At the same time, memory usage for storing graph structures has been reduced by up to 56% in heterogeneous graphs like ogbn-papers.

## Example model development lifecycle for GraphStorm on SageMaker

Figure 1: GraphStorm SageMaker architecture.

A common model development process is to perform model exploration locally on a subset of your full data, and once satisfied with the results train the full scale model. GraphStorm-SageMaker Pipelines allows you to do that by creating a  model pipeline you can execute locally to retrieve model metrics, and when ready execute your pipeline on the full data to produce models, predictions and graph embeddings for downstream tasks. In the next section you'll learn how to set up such pipelines for GraphStorm.

## Set up environment for SageMaker distributed training

You'll be using SageMaker Bring-Your-Own-Container (BYOC) to launch processing and training jobs. You need to create a PyTorch Docker image for distributed training, and we'll use the same image to process and prepare the graph for training.
You will use SageMaker Pipelines to automate jobs needed for GNN training. As a prerequisite, you'll need to have access to a [SageMaker Domain](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html) to access [SageMaker Studio](https://aws.amazon.com/sagemaker-ai/studio/) and [SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html).

### Create a SageMaker Domain

In order to use SageMaker Studio you will need to have a SageMaker Domain available. If you don't have one already, follow the steps in the [quick setup](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html) to create one:

1. Sign in to the [SageMaker AI console](https://console.aws.amazon.com/sagemaker/).
2. Open the left navigation pane.
3. Under **Admin configurations**, choose **Domains**.
4. Choose **Create domain**.
5. Choose **Set up for single user (Quick setup**). Your domain and user profile are created automatically.

### Set up appropriate roles to use with SageMaker Pipelines

To set up the SageMaker Pipelines you will need permissions to create ECR repositories, pull and push docker images to them, pull images from the AWS ECR Public Gallery, launch SageMaker jobs, manage SageMaker Pipelines, and interact with data on S3. We will create a role for Amazon EC2 on the AWS console, which will also create an associated instance profile to use with an EC2 instance.

You will also need access to a [SageMaker execution role](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) that your jobs assume during execution.
You can use the [Amazon SageMaker Role Manager](https://docs.aws.amazon.com/sagemaker/latest/dg/role-manager.html) to streamline the creation of the necessary roles.


### Set up the pipeline management environment

For this example we recommend you to set up a new EC2 instance with at least 300 GByte of disk space.
To set up an EC2 instance with the appropriate environment:


1. Launch an EC2 instance:

```bash
# Use an Ubuntu PyTorch 2.4.0 DLAMI (Ubuntu 22.04)
aws ec2 run-instances \
    --image-id "ami-0907e5206d941612f" \
    --instance-type "m6in.4xlarge" \
    --key-name my-key-name \
    --block-device-mappings '[{
        "DeviceName": "/dev/sda1",
        "Ebs": {
            "VolumeSize": 300,
            "VolumeType": "gp3",
            "DeleteOnTermination": true
        }
    }]'
```

This command creates an instance using the "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.4.1 (Ubuntu 22.04) 20241116" AMI, in the default VPC with the default security group. Make your instance accessible through SSH, using an appropriate security group or the [AWS Systems Session Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html), and log in to the instance.  You can also use the [AWS Console](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/tutorial-launch-my-first-ec2-instance.html) to create a new EC2 instance.

> NOTE: You may need to update the --image-id to the latest available. See https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html for instructions on finding the latest DLAMI.

Once logged in, you can set up your Python environment to run GraphStorm

```bash
conda init
eval $SHELL
# Available on the DLAMI, otherwise create a new conda env
conda activate pytorch

# Install dependencies
pip install sagemaker[local] boto3 ogb pyarrow

# Clone the GraphStorm repository to access the example code
git clone https://github.com/awslabs/graphstorm.git ~/graphstorm
```

### Download and prepare datasets

The Open Graph Benchmark (OGB) project hosts a number of graph datasets that can be used to benchmark the performance of graph learning systems. In this example you will use two citation network datasets, the ogbn-arxiv dataset for a small-scale demo, and the ogbn-papers100M dataset for a demonstration of GraphStorm's large-scale learning capabilities.

Because the two datasets have similar schemas and the same task (node classification) they allow us to emulate a typical data science pipeline, where we first do some model development and testing on a smaller dataset locally, and once ready launch SageMaker jobs to train on the full-scale data.


#### Prepare the ogbn-arxiv dataset

You'll download the smaller-scale [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) dataset to run a local test before launching larger scale SageMaker jobs on AWS. This dataset has ~170K nodes and ~1.2M edges.  You will use the following script to download the arxiv data and prepare them for GraphStorm.


```bash
# Provide the S3 bucket to use for output
BUCKET_NAME=<your-s3-bucket>
```

You will use this script to directly download, transform and upload the data to S3:

```bash
cd ~/graphstorm/examples/sagemaker-pipelines-graphbolt
python convert_arxiv_to_gconstruct.py \
    --output-prefix s3://$BUCKET_NAME/ogb-arxiv-input
```

This will create the tabular graph data on S3 which you can verify by running


```bash
aws s3 ls s3://$BUCKET_NAME/ogb-arxiv-input/
                           PRE edges/
                           PRE nodes/
                           PRE splits/
XXXX-XX-XX XX:XX:XX       1269 gconstruct_config_arxiv.json
```

Finally you'll also upload GraphStorm training configuration files for arxiv to use for training and inference

```bash
# Upload the training configurations to S3
aws s3 cp ~/graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml \
    s3://$BUCKET_NAME/yaml/arxiv_nc_train.yaml
aws s3 cp ~/graphstorm/inference_scripts/np_infer/arxiv_nc.yaml \
    s3://$BUCKET_NAME/yaml/arxiv_nc_inference.yaml
```

**Prepare the ogbn-papers100M dataset on SageMaker**

The papers-100M dataset is a large-scale graph dataset, with 111M nodes and ~3.2B edges when we add reverse edges. The data size is ~57GB so to make efficient use of our AWS resources we'll download and unzip the data in parallel, using multiple threads and upload directly to S3. To do so we will use the [axel](https://github.com/axel-download-accelerator/axel) and [ripunzip](https://github.com/google/ripunzip/) libraries.

You can either run this job as a SageMaker processing job or you can run the processing locally in the background while you work on building the GraphStorm Docker image and training a local model for the ogbn-arxiv dataset.

To run this process as a SageMaker Processing step, follow the steps below. You can launch and let the job execute in the background while proceeding through the rest of the steps, you can come back to this dataset later.


```bash
# Navigate to the example code and ensure Docker is installed
cd ~/graphstorm/examples/sagemaker-pipelines-graphbolt
sudo apt update
sudo apt install Docker.io
docker -v

# Build and push a Docker image to download and process the papers100M data
bash build_and_push_papers100M_image.sh
# This creates an ECR repository at
# $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/papers100m-processor

# Run a SageMaker job to do the processing and upload the output to S3
SAGEMAKER_EXECUTION_ROLE_ARN=<your-sagemaker-execution-role-arn>
ACCOUNT_ID=<your-aws-account-id>
REGION=us-east-1

aws configure set region $REGION
python sagemaker_convert_papers100m.py \
    --output-bucket $BUCKET_NAME \
    --execution-role-arn $SAGEMAKER_EXECUTION_ROLE_ARN \
    --region $REGION \
    --instance-type ml.m5.4xlarge \
    --image-uri  $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/papers100m-processor
```

This will produce the processed data at `s3://$BUCKET_NAME/ogb-papers100M-input`  which can then be used as input to GraphStorm.

> NOTE: Ensure your instance IAM profile is allow to perform `iam:GetRole` and `iam:GetPolicy` on your `SAGEMAKER_EXECUTION_ROLE_ARN`.


#### [Optional] Prepare the ogbn-papers100M dataset locally

If you prefer to pre-process the data locally, you can use the commands below on an Ubuntu 22.04 instance.

```bash
# Install axel for parallel downloads
sudo apt update
sudo apt -y install axel

# Download and install ripunzip for parallel unzipping
curl -L -O https://github.com/google/ripunzip/releases/download/v2.0.0/ripunzip_2.0.0-1_amd64.deb
sudo apt install -y ./ripunzip_2.0.0-1_amd64.deb

# Download and unzip data using multiple threads, this will take 10-20 minutes
mkdir ~/papers100M-raw-data
cd ~/papers100M-raw-data
axel -n 16 http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip
ripuznip unzip-file papers100M-bin.zip
cd papers100M-bin/raw
ripunzip unzip-file data.npz && rm data.npz

# Install process script dependencies
python -m pip install \
    numpy==1.26.4 \
    psutil==6.1.0 \
    pyarrow==18.1.0 \
    tqdm==4.67.1 \
    tqdm-loggable==0.2


# Process and upload to S3, this will take around 20 minutes
cd ~/graphstorm/examples/sagemaker-pipelines-graphbolt
python convert_ogb_papers100m_to_gconstruct.py \
    --input-dir ~/papers100M-raw-data
    --output-dir s3://$BUCKET_NAME/ogb-papers100M-input
```

### Build a GraphStorm Docker Image

Next you will build and push the GraphStorm PyTorch Docker image that you'll use to run the graph construction, training and inference jobs. If you have the papers-100M data downloading in the background, open a new terminal to build and push the GraphStorm image.


```bash
# Ensure Docker is installed
sudo apt update
sudo apt install -y Docker.io
docker -v

cd ~/graphstorm

bash ./docker/build_graphstorm_image.sh --environment sagemaker --device cpu

bash docker/push_graphstorm_image.sh -e sagemaker -r $REGION -a $ACCOUNT_ID -d cpu
# This will push an image to
# ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/graphstorm:sagemaker-cpu
```

Next, you will create a SageMaker Pipeline to run the jobs that are necessary to train GNN models with GraphStorm.

## Create SageMaker Pipeline

In this section, you will create a [Sagemaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-overview.html) on AWS SageMaker. The pipeline will run the following jobs in sequence:

* Launch GConstruct Processing job. This prepares and partitions the data for distributed training..
* Launch GraphStorm Training Job. This will train the model and create model output on S3.
* Launch GraphStorm Inference Job. This will generate predictions and embeddings for every node in the input graph.

```bash
PIPELINE_NAME="ogbn-arxiv-gs-pipeline"

bash deploy_papers100M_pipeline.sh \
    --account $ACCOUNT_ID \
    --bucket-name $BUCKET_NAME --role $SAGEMAKER_EXECUTION_ROLE_ARN \
    --pipeline-name $PIPELINE_NAME \
    --use-graphbolt false
```

### Inspect pipeline

Running the above will create a SageMaker Pipeline configured to run 3 SageMaker jobs in sequence:

* A GConstruct job that converts the tabular file input to a binary partitioned graph on S3.
* A GraphStorm training job that trains a node classification model and saves the model to S3.
* A GraphStorm inference job that produces predictions for all nodes in the test set, and creates embeddings for all nodes.

To review the pipeline, navigate to [SageMaker AI Studio](https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/studio-landing) on the AWS Console, select the domain and user profile you used to create the pipeline in the drop-down menus on the top right, then select **Open Studio**.

On the left navigation menu, select **Pipelines**. There should be a pipeline named **ogbn-arxiv-gs-pipeline**. Select that, which will take you to the **Executions** tab for the pipeline. Select **Graph** to view the pipeline steps.

### Execute SageMaker pipeline locally for ogbn-arxiv

The ogbn-arxiv data are small enough that you can execute the pipeline locally. Execute the following command to start a local execution of the pipeline:


```bash
PIPELINE_NAME="ogbn-arxiv-gs-pipeline"

python ~/graphstorm/sagemaker/pipeline/execute_sm_pipeline.py \
    --pipeline-name $PIPELINE_NAME \
    --region us-east-1 \
    --local-execution | tee arxiv-local-logs.txt
```

Note that we save the log output to `arxiv-local-logs.txt`. We'll use that later to analyze the training speed.

Once the pipeline finishes it will print a message like

```
Pipeline execution 655b9357-xxx-xxx-xxx-4fc691fcce94 SUCCEEDED
```

You can inspect its output on S3. Every pipeline execution will be under the prefix `s3://$BUCKET_NAME/pipelines-output/ogbn-arxiv-gs-pipeline/`

Every pipeline execution that shares the same input arguments will be under a randomized execution-identifying output path.
Note that the particular execution subpath might be different in your case.

```bash
aws s3 ls  s3://$BUCKET_NAME/pipelines-output/ogbn-arxiv-gs-pipeline/

# 761a4ff194198d49469a3bb223d5f26e

# There should only be one execution subpath, copy that into a new env variable
EXECUTION_SUBPATH="761a4ff194198d49469a3bb223d5f26e"
aws s3 ls \
    s3://$BUCKET_NAME/pipelines-output/ogbn-arxiv-gs-pipeline/$EXECUTION_SUBPATH/

# You will see the top-level outputs
# gconstruct/
# inference/
# model/

# gconstruct/ output
aws s3 ls \
    s3://$BUCKET_NAME/pipelines-output/ogbn-arxiv-gs-pipeline/$EXECUTION_SUBPATH/gconstruct/

# We get the 2 graph partitions (part0, part1) and metadata JSON files that describe the graph
# data_transform_new.json  edge_label_stats.json  edge_mapping.pt  node_label_stats.json  node_mapping.pt  ogbn-arxiv.json  part0  part1

# model/ output
aws s3 ls \
    s3://$BUCKET_NAME/pipelines-output/ogbn-arxiv-gs-pipeline/$EXECUTION_SUBPATH/model/

# We get a model snapshot for every epoch
# epoch-0  epoch-1  epoch-2  epoch-3  epoch-4  epoch-5  epoch-6  epoch-7  epoch-8  epoch-9

# inference/ output
aws s3 ls \
    s3://$BUCKET_NAME/pipelines-output/ogbn-arxiv-gs-pipeline/$EXECUTION_SUBPATH/inference/

# We get two prefixes, one containing the embeddings and one the predictions
# embeddings/  predictions/

```

You'll be able to see the output of each step in the pipeline. The GConstruct job created the partitioned graph, the training job created models for 10 epochs, and the inference job created embeddings for the nodes and predictions for the nodes in the test set.

You can inspect the mean epoch and evaluation time using the provided `analyze_training_time.py` script and the log file you created:


```bash
python analyze_training_time.py --log-file arxiv-local-logs.txt

Reading logs from file: arxiv-logs.txt

=== Training Epochs Summary ===
Total epochs completed: 10
Average epoch time: 7.43 seconds

=== Evaluation Summary ===
Total evaluations: 11
Average evaluation time: 2.25 seconds
```

Note that these numbers will vary depending on your instance type.

### Create GraphBolt Pipeline

Now that you have established a baseline for performance you can create another pipeline that uses the GraphBolt graph representation to compare the performance.

You can use the same pipeline creation script, but change two variables, providing a new pipeline name, and setting `USE_GRAPHBOLT` to `“true”` as `--use-graphbolt true`.


```bash
# Deploy the GraphBolt-enabled pipeline
PIPELINE_NAME_GRAPHBOLT="ogbn-arxiv-gs-graphbolt-pipeline"
BUCKET_NAME="my-s3-bucket"
bash deploy_arxiv_pipeline.sh \
    --account "<aws-account-id>" \
    --bucket-name $BUCKET_NAME --role "<execution-role>" \
    --pipeline-name $PIPELINE_NAME_GRAPHBOLT \
    --use-graphbolt true
# Execute the pipeline locally
python ~/graphstorm/sagemaker/pipeline/execute_sm_pipeline.py \
    --pipeline-name $PIPELINE_NAME_GRAPHBOLT \
    --region us-east-1 \
    --local-execution | tee arxiv-local-gb-logs.txt
```

Analyzing the training logs you can see a noticeable reduction in per-epoch time:

```bash
python analyze_training_time.py --log-file arxiv-local-gb-logs.txt

Reading logs from file: arxiv-gb-logs.txt

=== Training Epochs Summary ===
Total epochs completed: 10
Average epoch time: 6.83 seconds

=== Evaluation Summary ===
Total evaluations: 11
Average evaluation time: 1.99 seconds
```

For such a small graph the performance gains are modest, around 13% per epoch time. Moving on to large data however, the potential gains are much larger. In the next section you will create a pipeline and train a model for `papers-100M`, a citation graph with 111M nodes and 3.2B edges.

## Create SageMaker Pipeline for distributed training

Once the papers-100M data have finished processing and exist on S3, either through your local job or the SageMaker Processing job, you can set up a pipeline to train a model on that dataset.

### Build the GraphStorm GPU image

For this job you will use large GPU instances, so you will build and push the GPU image this time:


```bash
cd ~/graphstorm

bash ./docker/build_graphstorm_image.sh --environment sagemaker --device gpu

bash docker/push_graphstorm_image.sh -e sagemaker -r $REGION -a $ACCOUNT_ID -d gpu
```

### Deploy and execute pipelines for papers-100M

Before you deploy your new pipeline, upload the training YAML configuration for papers-100M to S3:


```bash
aws s3 cp \
    ~/graphstorm/training_scripts/gsgnn_np/papers_100M_nc.yaml \
    s3://$BUCKET_NAME/yaml/
```


Now you are ready to deploy your initial pipeline for papers-100M

```bash
PIPELINE_NAME="ogb-papers100M-pipeline"
cd ~/graphstorm/examples/sagemaker-pipelines-graphbolt/
bash deploy_papers100M_pipeline.sh \
    --account <aws-account-id> \
    --bucket-name <s3-bucket> --role <execution-role> \
    --pipeline-name $PIPELINE_NAME \
    --use-graphbolt false
```

Execute the pipeline and let it run the background.

```bash
python ~/graphstorm/sagemaker/pipeline/execute_sm_pipeline.py \
    --pipeline-name $PIPELINE_NAME \
    --region us-east-1
    --async-execution
```

>Note that your account needs to meet the required quotas for the requested instances. Here the defaults are set to four `ml.g5.48xlarge` for training jobs and one `ml.r5.24xlarge` instance for a processing job. To adjust your SageMaker service quotas you can use the [Service Quotas console UI](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas).  To run both pipelines in parallel you will need 8 x $TRAIN_GPU_INSTANCE and 2 x $GCONSTRUCT_INSTANCE.


Next, you can deploy and execute another pipeline, now with GraphBolt enabled:

```bash
PIPELINE_NAME_GRAPHBOLT="ogb-papers100M-graphbolt-pipeline"
bash deploy_papers100M_pipeline.sh \
    --account <aws-account-id> \
    --bucket-name <s3-bucket> --role <execution-role> \
    --pipeline-name $PIPELINE_NAME_GRAPHBOLT \
    --use-graphbolt true

# Execute the GraphBolt-enabled pipeline on SageMaker
python ~/graphstorm/sagemaker/pipeline/execute_sm_pipeline.py \
    --pipeline-name $PIPELINE_NAME_GRAPHBOLT \
    --region us-east-1 \
    --async-execution
```

### Compare performance for GraphBolt-enabled training

Once both pipelines have finished executing, which should take approximately 4 hours, you can compare the training times for both cases. To do so you will need to find the pipeline execution display names for the two papers-100M pipelines.

The easiest way to do so is through the Studio pipeline interface. In the Pipelines page you visited previously, there should be two new  pipelines named **ogb-papers100M-pipeline** and **ogb-papers100M-graphbolt-pipeline**. Select **ogb-papers100M-pipeline**, which will take you to the **Executions** tab for the pipeline. Copy the name of the latest successful execution and use that to run the training analysis script:


```bash
python analyze_training_time.py \
    --pipeline-name $PIPELINE_NAME \
    --execution-name execution-1734404366941
```

Your output will look like

```bash
== Training Epochs Summary ===
Total epochs completed: 15
Average epoch time: 73.95 seconds

=== Evaluation Summary ===
Total evaluations: 15
Average evaluation time: 15.07 seconds
```

Now do the same for the GraphBolt-enabled pipeline:

```bash
python analyze_training_time.py \
    --pipeline-name $PIPELINE_NAME_GRAPHBOLT \
    --execution-name execution-1734463209078
```

You will see the improved per-epoch and evaluation times:

```bash
== Training Epochs Summary ===
Total epochs completed: 15
Average epoch time: 54.54 seconds

=== Evaluation Summary ===
Total evaluations: 15
Average evaluation time: 4.13 seconds
```

Without loss in accuracy, the latest version of GraphStorm achieved a **~1.4x speedup per epoch, and a 3.6x speedup in evaluation time!**

We encourage you to try out GraphStorm with GraphBolt enabled to see how it can benefit your large-scale graph learning use cases.

[1] DGL team GraphBolt benchmarks: https://www.dgl.ai/release/2024/03/06/release.html
