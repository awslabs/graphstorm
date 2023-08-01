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
python3 gen_mag_dataset.py --savepath /tmp/ogbn_arxiv \
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
