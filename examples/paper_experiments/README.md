This folder contains the data processing and code to profile GraphStorm framework on Amazon Review dataset.

# Data Processing
## Amazon Review
### Process raw dataset into parquet files
We first download the full Amazon Review dataset from https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/.
Review data is in https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/All_Amazon_Review.json.gz
Meta data is in https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles/All_Amazon_Meta.json.gz
```
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz \
-P raw_data/
```

After you unzip the above two files, run the following two scripts to generate the input for the graph construction step.

```
python construct_AR_items.py
python construct_AR_reviews.py
```

### Run gconstruct on process_data
After the first step, the processed data are under processed_data/amazon_review/
Once the data are processed, run the following command to construct a graph
for node classification and link prediction tasks.
Recommended machine: x1e.32xlarge (https://aws.amazon.com/ec2/instance-types/x1e/)
Runing time: 8hours14minutes Peak Memory: 1020GB

The command takes `AR_construction_full` that specifies the input data for graph
construction, constructs the graph, and saves the parition to `amazon_review`.

```
python -m graphstorm.gconstruct.construct_graph \
           --conf-file gconstruct_configs/AR_construction_full.json \
           --output-dir datasets/amazon_review_full --graph-name amazon_review \
           --num-processes 16 --num-parts 4  --logging-level debug \
           --skip-nonexist-edges --add-reverse-edges
```
The processed data is stored under datasets/amazon_review_full/.
The script partitions the graph into four parts.

### Run graphstorm training
By default, we use four instances for training and their ips are written in ip_list.txt.
Please refer to GraphStorm Distributed Training Tutorials for more details
(https://graphstorm.readthedocs.io/en/latest/scale/distributed.html).

Node classification
```
python -m graphstorm.run.gs_node_classification --num-trainers 8 \
    --num-servers 1 --num-samplers 0 --part-config datasets/amazon_review_full/amazon_review.json \
    --ip-config ip_list.txt --ssh-port 22 --cf training_configs/main_experiments/AR_nc_gnn.yaml
```

Link prediction
```
python -m graphstorm.run.gs_link_prediction --num-trainers 8 \
    --num-servers 1 --num-samplers 0 --part-config datasets/amazon_review_full/amazon_review.json \
    --ip-config ip_list.txt --ssh-port 22 --cf training_configs/main_experiments/AR_lp_gnn.yaml
```

### Performances
In this section, we list the correspoding configs of each experiment and its performance.
By default, we use accuracy/MRR as node classification/link prediction metrics, respectively.
| Config | Description | Performance |
|----------|----------|----------|
| main_experiments/AR_lp_ft.yaml    |   fine-tune BERT+GNN for link prediction on amazon review      |  0.9710        |
| main_experiments/AR_lp_gnn.yaml    |  pre-trained BERT+GNN for link prediction on amazon review       |   0.9602       |
| main_experiments/AR_nc_ft.yaml     |  fine-tune BERT+GNN for node classification on amazon review       |  0.8963        |
| main_experiments/AR_nc_gnn.yaml     |  pre-trained BERT+GNN for node classification on amazon review        |  0.8407        |
| main_experiments/mag_lp_ft.yaml    |  fine-tune BERT+GNN for link prediction on MAG        |   0.6841       |
| main_experiments/mag_lp_gnn.yaml    |  pre-trained BERT+GNN for link prediction on MAG         |  0.4873        |
| main_experiments/mag_nc_ft.yaml    |  fine-tune BERT+GNN for node classification on MAG        |  0.6333        |
| main_experiments/mag_nc_gnn.yaml    |  pre-trained BERT+GNN for node classification on MAG        |  0.5715        |

In the following table, we show the performance of link prediction on the Amazon Review dataset with different settings. (Test with GraphStorm 0.2.1 on 4 g5.48xlarge instances)
| Config | Description | Performance (MRR) | epoch time (s) |
| link_prediction/lp_also_buy_1_layer_rgcn_dot_contrastive_loss_in_batch.yaml | In batch negative sampling with contrastive loss | 0.951 | 1340.90 |
| link_prediction/lp_also_buy_1_layer_rgcn_dot_contrastive_loss_joint_1024_negative.yaml | Joint negative sampling with batch size of 1024 with contrastive loss | 0.956 | 1344.65 |
| link_prediction/lp_also_buy_1_layer_rgcn_dot_contrastive_loss_joint_32_negative.yaml | Joint negative sampling with batch size of 32 with contrastive loss | 0.958 | 1286.64 |
| link_prediction/lp_also_buy_1_layer_rgcn_dot_contrastive_loss_joint_4_negative.yaml | Joint negative sampling with batch size of 4 with contrastive loss | 0.956 | 1289.9 |
| link_prediction/lp_also_buy_1_layer_rgcn_dot_contrastive_loss_uniform_32_negative.yaml | Uniform negative sampling with batch size of 32 with contrastive loss | 0.957 | 1726.19 |

| link_prediction/lp_also_buy_1_layer_rgcn_dot_entropy_loss_in_batch.yaml | In batch negative sampling with cross entropy loss | 0.250 | 1343.94 |
| link_prediction/lp_also_buy_1_layer_rgcn_dot_entropy_loss_joint_1024_negative.yaml| Joint negative sampling with batch size of 1024 with cross entropy loss | 0.334 | 1330.50 |
| link_prediction/lp_also_buy_1_layer_rgcn_dot_entropy_loss_joint_32_negative.yaml | Joint negative sampling with batch size of 32 with cross entropy loss | 0.380 | 1290.72 |
| link_prediction/lp_also_buy_1_layer_rgcn_dot_entropy_loss_joint_4_negative.yaml | Joint negative sampling with batch size of 4 with cross entropy loss | 0.645 | 1288.53 |
| link_prediction/lp_also_buy_1_layer_rgcn_dot_entropy_uniform_32_negative.yaml | Uniform negative sampling with batch size of 32 with cross entropy loss | 0.377 | 1746.68 |
