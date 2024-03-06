This folder contains the data processing and code to reproduce the graphstorm paper results.

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

After you unzip the above two files, we generate the parquet files as input of graph construction step. 
```
python construct_AR_items.py
python construct_AR_reviews.py
```

### Run gconstruct on process_data
In the first step, the processed data are under processed_data/amazon_review/
Once the data are processed, run the following command to construct a graph
for node classification and link prediction tasks. 
Recommended machine: x1e.32xlarge	

The command takes `AR_Video_Games.json` that specifies the input data for graph 
construction, constructs the graph, and saves the parition to `amazon_review`.

```
python -m graphstorm.gconstruct.construct_graph \
           --conf-file gconstruct_configs/AR_construction_full.json \
           --output-dir datasets/amazon_review_full --graph-name amazon_review \
           --num-processes 16 --num-parts 4  --logging-level debug \
           --skip-nonexist-edges --add-reverse-edges
```
The processed data is stored under datasets/amazon_review_full/

### Run graphstorm training 
By default, we use four instances for training and their ips are written in ip_list.txt

Node classification
```
python -m graphstorm.run.gs_node_classification --num-trainers 8 \
    --num-servers 1 --num-samplers 0 --part-config datasets/amazon_review_full/amazon_review.json \
    --ip-config ip_list.txt --ssh-port 22 --cf training_configs/AR_nc_gnn.yaml
```

Link prediction
```
python -m graphstorm.run.gs_link_prediction --num-trainers 8 \
    --num-servers 1 --num-samplers 0 --part-config datasets/amazon_review_full/amazon_review.json \
    --ip-config ip_list.txt --ssh-port 22 --cf training_configs/AR_lp_gnn.yaml
```
