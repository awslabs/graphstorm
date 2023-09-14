# GraphStorm Tools

GraphStorm provides a set of scripts to help users process graph data and run experiments.

**Note**: all scripts should be ran within the GraphStorm Docker container environment.

## Generate GraphStorm built-in datasets
GraphStorm has a set of built-in datasets that can help users to quickly learn the usage of GraphStorm. These datasets include:
- OGBN arxiv, GraphStorm dataset name: "**ogbn-arxiv**";
- OGBN products, GraphStorm dataset name: "**ogbn-products**";
- OGBN papers100M, GraphStorm dataset name: "**ogbn-papers100M**";
- OGBN mag, GraphStorm dataset name: "**ogbn-mag**".

Users can use the `gen_ogb_dataset.py` and `gen_mag_dataset.py` script to automatically download the four graph datasets from OGBN site and process them into DGL graph format.

Usage examples:

For OGBN arxiv, products, and papers100M graphs, users need to sepecify the value for the `--dataset` argument.
```bash
python3 gen_ogb_dataset.py --savepath /tmp/ogbn_arxiv \
                           --dataset ogbn-axiv \
                           --edge-pct 0.8
```
Because the OGBN mag is a heterogeneous graph, it has a separated script.
```bash
python3 gen_mag_dataset.py --savepath /tmp/ogbn-mag \
                           --edge-pct 0.8
```

For other arugments used in the two scripts, please read the **main()** method.

## Kill and cleanup GraphStorm running processes
If users set GraphStrom running processes as deamons, you can use the `kill_cleanup_disttrain.sh` to stop these processess and release computation resource.

Usage examples:

```bash
bash ./kill_cleanup_disttrain.sh /tmp/ip_list.txt ogbn-arxiv 2222
```

The `kill_cleanup_disttrain.sh` command require three arguments.
- **IP list file**: Required, the IP address list file used when laucn GraphStorm run scripts. Suggest to use absolute path to avoid path not found issue.
- **graph name**: Required, the graph dataset name specified in the configuration file.
- **port number**: Optional, the communication port number used when launch GraphStorm run scripts. If run within GraphStorm docker environment, no need to provided. Default value is 2222.

## Split graph data into the input format GraphStorm required
The input format of GraphStorm is a DistDGL graph. GraphStorm provides two scripts to help user split their graph, which should be a DGL graph, into the required input format for different tasks.

The `partition_graph.py` can split graph for the tasks of Node Classification, Node Regression, Edge Classification, and Edge Regression. The `partition_graph_lp.py` can split graph for the task of Link Prediction.

Usage examples:

Below command can download the OGBN arxiv data, process it into DGL graph, and finally split it into two partition. The partitioned graph is save at the /tmp/ogbn_arxiv_nc_2p folder.
```bash
python3 /graphstorm/tools/partition_graph.py --dataset ogbn-arxiv \
                                             --filepath /tmp/ogbn-arxiv-nc/ \
                                             --num-parts 2 \
                                             --output /tmp/ogbn_arxiv_nc_2p
                                             --bert-name "allenai/scibert_scivocab_uncased"
```

Below command can download the OGBN mag data, process it into DGL graph, and finally split it into two partition. The partitioned graph is save at the /tmp/ogbn_mag_lp_2p folder.
```bash
python3 /graphstorm/tools/partition_graph_lp.py --dataset ogbn-mag \
                                                --filepath /tmp/ogbn-mag-lp/ \
                                                --num-parts 2 \
                                                --target-etypes author,writes,paper \
                                                --output /tmp/ogbn_mag_lp_2p
```

For details of the arguments of the two scripts, please refer to their **main()** funciton.

Below command can generate the homogeneous version of ogbn-arxiv. The partitioned graph is save at the /tmp/ogbn_arxiv_nc_2p folder.
```bash
python3 /graphstorm/tools/gen_ogb_dataset.py --savepath /tmp/ogbn-arxiv-nc/  \
                          --dataset ogbn-arxiv \
                          --retain-original-features true \
                          --is-homo
                           
python3 /graphstorm/tools/partition_graph.py --dataset ogbn-arxiv \
                                             --filepath /tmp/ogbn-arxiv-nc/ \
                                             --num-parts 1 \
                                             --output /tmp/ogbn_arxiv_nc_train_val_1p_4t  \
                                             --is-homo     
```

## Convert node features from distDGL format to WholeGraph format

Use the `convert_feat_to_wholegraph.py` script with `--dataset-path` pointing to the distDGL folder of partitions. It will convert all the node features and labels to WholeGraph compatible format.
```
python3 convert_feat_to_wholegraph.py --dataset-path ogbn-mag240m-2p
```

The script will create a new folder '`wholegraph`' under '`ogbn-mag240m-2p`' containing the WholeGraph input files and will trim the distDGL file `node_feat.dgl` in each partition to remove the feature and label attributes, leaving only other attributes such as `train_mask`, `test_mask`, and `val_mask` intact. It also saves a backup `node_feat.dgl.bak`.

The features in those files can be loaded in memory via the WholeGraph API by giving the folder path and feature prefix (`<node_type>~<feat_name>`).
Below is an example showing how to load the data:
```python
import json
import os
import torch
import pylibwholegraph.binding.wholememory_binding as wmb
import pylibwholegraph.torch as wgth

with open('ogbn-mag240m/wholegraph/metadata.json') as f:
    metadata = json.load(f)

torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
wmb.init(0)
torch.distributed.init_process_group('nccl')
global_comm = wgth.comm.get_global_communicator()

cache_policy = wgth.create_builtin_cache_policy(
    "none", # cache type
    "distributed",
    "cpu",
    "readonly", # access type
    0.0, # cache ratio
)

paper_feat_wg = wgth.create_embedding(
                    global_comm,
                    'distributed',
                    'cpu',
                    getattr(torch, metadata['paper/feat']['dtype'].split('.')[1]),
                    metadata['paper/feat']['shape'],
                    optimizer=None,
                    cache_policy=cache_policy,
                )
# 'part_count' is the number of partition files. For distDGL it will always be the number of machines.
paper_feat_wg.get_embedding_tensor().from_file_prefix('ogbn-mag240m/wholegraph/paper~feat', part_count=4)
```

