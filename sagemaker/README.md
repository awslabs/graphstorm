# Running GraphStorm jobs on Amazon SageMaker
This submodule provides support to run GraphStorm graph construction, partitioning, training and inference on Amazon SageMaker.
To build a SageMaker compatible Docker image, please refer to [Build GraphStorm SageMaker docker image] (https://github.com/awslabs/graphstorm/docker/sagemaker).


## Launch SageMaker tasks
Use scripts under graphstorm.sagemaker.launch to launch SageMaker tasks.
Please make sure you already setup your SageMaker environment.
Please refer to [Amazon SageMaker service](https://aws.amazon.com/pm/sagemaker) for how to get access to Amazon SageMaker.

### Common task arguments

All SageMaker tasks share some common arguments that can be found in `<gs-root>/sagemaker/launch/common_parser.py`

* `image_url`: (required) The URI to the GraphStorm SageMaker image on ECR.
* `role`: (required) The SageMaker execution role that will be used to run the job.
* `instance-type`: The SageMaker instance type used to run the job.
* `instance-count`: The number of SageMaker instances used to run the job.
    For GConstruct jobs, the number is always 1.
* `region`: The region in which we will launch the job. Default is `us-west-2`, but ensure it matches the
    region of your image.
* `task-name`: A user-defined task name for the job.
* `sm-estimator-parameters`: Parameters that will be forwarded to the SageMaker Estimator/Processor.
    See the section `Passing additional arguments to the SageMaker Estimator/Processor` for details.
* `async-exection`: When this flag is set the job will run in async mode and return immediatelly.
    Otherwise the job will block and logs will be printed to the console.

### Launch GraphStorm graph construction using Amazon SageMaker service

#### Preparing the example dataset
We use the built-in acm dataset as an example.
You can generate the raw acm dataset in parquet format using the following instructions
(See [Use Your Own Data Tutorial](https://github.com/awslabs/graphstorm/wiki/tutorials-own-data#use-own-data) for more details):

```bash
cd ~/
git clone https://github.com/awslabs/graphstorm.git
GS_HOME=~/graphstorm/
cd $GS_HOME/examples
python3 acm_data.py --output-path acm_raw
```
The raw graph input data will be stored at ~/graphstorm/examples/acm_raw. The input configuration JSON is also generated and stored in the same path at  ~/graphstorm/examples/acm_raw/config.json.

#### Launch graph processing task
Before launching the task, you need to upload the raw acm dataset (i.e., /tmp/acm_raw) into S3.
```bash
aws s3 cp --recursive ~/graphstorm/examples/acm_raw s3://PATH_TO/acm/acm_raw/
```

Then, you can use the following command to launch a SageMaker graph construction task. For `graph-config-file`, we also accept configurations designed for distributed processing.

```bash
python3 launch/launch_gconstruct.py \
        --image-url <AMAZON_ECR_IMAGE_URI> \
        --region <region> \
        --entry-point run/gconstruct_entry.py \
        --role <ROLE_ARN> \
        --input-graph-s3 s3://PATH_TO/acm/ \
        --output-graph-s3 s3://PATH_TO/acm_output/ \
        --graph-name acm \
        --graph-config-file acm_raw/config.json \
        --sm-estimator-parameters "volume_size_in_gb=10"
```
The processed graph data is stored at s3://PATH_TO/acm_output/.

Note: The `--input-graph-s3` path will be mapped into `/opt/ml/processing/input` and used as the command working directory when launching the graph construction command.
The graph configuration file should be stored with the input graph data.
The argument `--graph-config-file` should be a relative path to `--input-graph-s3`.

### Launch graph partitioning task

If your data are in the [DGL chunked format](https://docs.dgl.ai/guide/distributed-preprocessing.html#specification)
you can perform distributed partitioning using SageMaker to prepare your data for distributed training.

```bash
python launch/launch_partition.py --graph-data-s3 ${DATASET_S3_PATH} \
    --num-parts ${NUM_PARTITIONS} --instance-count ${NUM_PARTITIONS} \
    --output-data-s3 ${OUTPUT_PATH} --instance-type ${INSTANCE_TYPE} \
    --image-url ${IMAGE_URI} --region ${REGION} \
    --role ${ROLE}  --entry-point "run/partition_entry.py" \
    --metadata-filename ${METADATA_FILE} \
    --log-level INFO --partition-algorithm ${ALGORITHM}
```

Running the above will take as input the dataset in chunked format from `${DATASET_S3_PATH}`
and create a DistDGL graph with `${NUM_PARTITIONS}` under the output path,
`${OUTPUT_PATH}`. Currently we only support `random` as the partitioning algorithm.

### Launch GraphStorm training using Amazon SageMaker service

#### Preparing training data and training task config.

We use the built-in ogbn-arxiv dataset as an example.
First you need to partition the graph dataset by following the instructions:
```bash
cd ~/
git clone https://github.com/awslabs/graphstorm.git
GS_HOME=~/graphstorm/
python3 $GS_HOME/tools/partition_graph.py --dataset ogbn-arxiv \
                                          --filepath /tmp/ogbn-arxiv-nc/ \
                                          --num_parts 2 \
                                          --output /tmp/ogbn_arxiv_nc_2p
```
The partitioned graph will be stored at /tmp/ogbn_arxiv_nc_2p.

#### Launch train task
Before launching the task, you need to upload the partitioned graph (i.e., /tmp/ogbn_arxiv_nc_2p) into S3.
You also need to upload the yaml config file into S3.
You can find the example yaml file in https://github.com/awslabs/graphstorm/blob/main/training_scripts/gsgnn_np/arxiv_nc.yaml.
```bash
aws s3 cp --recursive /tmp/ogbn_arxiv_nc_2p s3://PATH_TO/ogbn_arxiv_nc_2p/
aws s3 cp PATH_TO/arxiv_nc.yaml s3://PATH_TO_TRAINING_CONFIG/arxiv_nc.yaml
```

Then, you can use the following command to launch a SageMaker training task.
```bash
cd $GS_HOME/sagemaker/
python3 launch/launch_train.py \
        --image-url <AMAZON_ECR_IMAGE_URI> \
        --region us-east-1 \
        --entry-point run/train_entry.py \
        --role <ROLE_ARN> \
        --graph-data-s3 s3://PATH_TO/ogbn_arxiv_nc_2p/ \
        --yaml-s3 s3://PATH_TO_TRAINING_CONFIG/arxiv_nc.yaml \
        --model-artifact-s3 s3://PATH_TO_SAVE_TRAINED_MODEL/ \
        --graph-name ogbn-arxiv \
        --task-type "node_classification" \
        --num-layers 1 \
        --hidden-size 128 \
        --backend gloo \
        --batch-size 128 \
        --node-feat-name node:feat
```
The trained model artifact will be stored in the S3 address provided through `--model-artifact-s3`.
You can use following command to check the model artifacts:
```
aws s3 ls s3://PATH_TO_SAVE_TRAINED_MODEL/
```

If you want to resume a saved model checkpoint to do model fine-tuning you can pass
the S3 address of the model checkpoint through the ``--model-checkpoint-to-load``
argument. For example by passing ``--model-checkpoint-to-load s3://mag-model/epoch-2/``,
GraphStorm will initialize the model parameters with the model checkpoint stored in ``s3://mag-model/epoch-2/``.

Please note `save_embed_path` and `save_prediction_path` must be disabled, i.e., set to 'None' when using SageMaker.
They only work with shared file system while SageMaker solution does not support using shared file system now.


### Launch inference task using built-in inference script
Inference task can use the same graph as training task. You can also run inference on a new graph.
In this example, we will use the same graph.

You can use the following command to launch a SageMaker offline inference task.
```bash
cd $GS_HOME/sagemaker/
python3 launch/launch_infer.py \
        --image-url <AMAZON_ECR_IMAGE_URI> \
        --region us-east-1 \
        --entry-point run/infer_entry.py \
        --role <ROLE_ARN> \
        --graph-data-s3 s3://PATH_TO/ogbn_arxiv_nc_2p/ \
        --yaml-s3 s3://PATH_TO_TRAINING_CONFIG/arxiv_nc.yaml \
        --model-artifact-s3  s3://PATH_TO_SAVED_MODEL/ \
        --output-emb-s3 s3://PATH_TO_SAVE_GENERATED_NODE_EMBEDDING/ \
        --output-prediction-s3 s3://PATH_TO_SAVE_PREDICTION_RESULTS \
        --graph-name ogbn-arxiv \
        --task-type "node_classification" \
        --num-layers 1 \
        --hidden-size 128 \
        --backend gloo \
        --batch-size 128 \
        --node-feat-name node:feat
```
The generated node embeddings will be uploaded into s3://PATH_TO_SAVE_GENERATED_NODE_EMBEDDING/.
The prediction results of node classification/regression or edge classification/regression tasks will be uploaded into s3://PATH_TO_SAVE_PREDICTION_RESULTS/
You can use following command to check the corresponding outputs:

```bash
aws s3 ls s3://PATH_TO_SAVE_GENERATED_NODE_EMBEDDING/
aws s3 ls s3://PATH_TO_SAVE_PREDICTION_RESULTS/
```

### Passing additional arguments to the SageMaker Estimator/Processor

Sometimes you might want to pass additional arguments to the constructor of the SageMaker
Estimator/Processor object we use to launch SageMaker tasks, e.g. to set a max runtime, or
set a VPC configuration. Our launch scripts support forwarding arguments to the base class
object through a `kwargs` dictionary.

To pass additional `kwargs` directly to the Estimator/Processor constructor, you can use
the `--sm-estimator-parameters` argument, providing a string of space-separated arguments
(enclosed in double quotes `"` to ensure correct parsing) and the
format `<argname>=<value>` for each argument.

`<argname>` needs to be a valid SageMaker Estimator/Processor
argument name and `<value>` a value that can be parsed as a Python literal, **without
spaces**.

For example, to pass a specific max runtime, subnet list, and enable inter-container
traffic encryption for a train, inference, or partition job you'd use:

```bash
python3 launch/launch_[infer|train|partition] \
    <other arugments> \
    --sm-estimator-parameters "max_run=3600 volume_size=100 encrypt_inter_container_traffic=True subnets=['subnet-1234','subnet-4567']"
```

Notice how we don't include any spaces in `['subnet-1234','subnet-4567']` to ensure correct parsing of the list.

The train, inference and partition scripts launch SageMaker Training jobs that rely on the `Estimator` base class:
For a full list of `Estimator` parameters see:
https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.EstimatorBase

The GConstruct job will launch a SageMaker Processing job that relies on the `Processor` base class, so its
arguments are different, e.g. `volume_size_in_gb` for the `Processor` vs. `volume_size` for the `Estimator`.

So the above example would become:

```bash
python3 launch/launch_gconstruct \
    <other arugments> \
    --sm-estimator-parameters "max_runtime_in_seconds=3600 volume_size_in_gb=100"
```


For a full list of `Processor` parameters see:
https://sagemaker.readthedocs.io/en/stable/api/training/processing.html
