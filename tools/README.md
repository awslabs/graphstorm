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
                                             --lm-model-name "allenai/scibert_scivocab_uncased"
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

## Use WholeGraph to accelerate training and inferencing
Graphstorm leverages NVIDIA’s [Wholegraph](https://github.com/rapidsai/wholegraph) framework to efficiently transfer node and edge features between machines. This capability can substantially enhance the speed of both training and inferencing pipelines. To take advantage of this feature, users are required to have EFA network support on their cluster. For a step-by-step setup guide, please refer to the [tutorial](https://graphstorm.readthedocs.io/en/latest/advanced/advanced-wholegraph.html). Converting node and edge features to the WholeGraph format is the only manual step; the rest of the process is seamless.

Please note, we do not support conversion of `train_mask`, `test_mask`, `val_mask` or `labels` to WholeGraph format. Make sure to convert all the node and edge features to WholeGraph format using `convert_feat_to_wholegraph.py` toolkit to utilize the framework.

#### Convert features from distDGL format to WholeGraph format

Use the `convert_feat_to_wholegraph.py` script with `--dataset-path` pointing to the distDGL folder of partitions. Use the argument `--node-feat-names` to specify the node features that should be converted to WholeGraph compatible format. Similarly, the `--edge-feat-names` allows you to specify the edge features that need to be transformed into a format suitable for WholeGraph. For example:

```
python3 convert_feat_to_wholegraph.py --dataset-path ogbn-mag240m-2p --node-feat-names paper:feat
```
or
```
python3 convert_feat_to_wholegraph.py --dataset-path dataset --node-feat-names paper:feat author:feat,feat2 institution:feat
```

The script will create a new folder '`wholegraph`' under '`ogbn-mag240m-2p`' containing the WholeGraph input files and will trim the distDGL file `node_feat.dgl` in each partition to remove the specified feature attributes, leaving only other attributes such as `train_mask`, `test_mask`, `val_mask` or  `labels` intact. It also saves a backup `node_feat.dgl.bak`.

Similarly, users can use  `--edge-feat-names` to convert edge features to WholeGraph compatible format.

```
python3 convert_feat_to_wholegraph.py --dataset-path ogbn-mag240m-2p --node-feat-names paper:feat --edge-feat-names author,writes,paper:feat
```

when `--edge-feat-names` is used, the  '`wholegraph`' folder will contain the edge features converted into WholeGraph format and will trim the distDGL file `edge_feat.dgl` in each partition to remove the specified feature attributes.

### Convert large features from distDGL format to WholeGraph format

The conversion script has a minimum memory requirement of 2X of the size of the input nodes and edge features in a graph. We offer a low-memory option that significantly reduces memory usage, requiring only 2X of the size of the largest node or edge feature in the graph, with the trade-off of longer conversion time. Users can enable this option by using the `--low-mem` argument.
```
python3 convert_feat_to_wholegraph.py --dataset-path ogbn-mag240m-2p --node-feat-names paper:feat --low-mem
```

## Do graph data sanity check
GraphStorm provides a tool to do graph feature and mask sanity check. Use `graph_sanity_check.py` script with `--dataset-path` pointing to the distDGL folder of partitions to check a partitioned graph data. By default, it will check whether any node feature or edge feature has `NaN` (Not a Number) or `Inf` (Infinite number) data. It will also check whether the features are normalized into the range of [-1, 1]. If not, it will print a warning. Use the argument `--node-masks` to specify the node masks to check and `--edge-masks` to specify the edge masks to check. The script will check whether GraphStorm can parse the mask without any error.

For example

```
>>> python3 graph_sanity_check.py --dataset-path /data/movie_lens_2p_example/ --node-masks user:train_mask,test_mask movie:val_mask --edge-masks  user,rating,movie:val_mask,train_mask

ERROR: [Node type: user][Feature Name: test2][Part part0]: There are NaN values in the feature, please check.
ERROR: [Node type: user][Feature Name: test2][Part part1]: There are NaN values in the feature, please check.
ERROR: [Node type: user][Feature Name: test][Part part0]: There are NaN values in the feature, please check.
ERROR: [Node type: user][Feature Name: test][Part part1]: There are NaN values in the feature, please check.
WARNING: [Node type: movie][Feature Name: label][Part part0]: There are some value out of the range of [-1, 1].It won't cause any error, but it is recommended to normalize the feature.
WARNING: [Node type: movie][Feature Name: label][Part part1]: There are some value out of the range of [-1, 1].It won't cause any error, but it is recommended to normalize the feature.
```
