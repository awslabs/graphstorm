This folder contains the data processing script to process the raw MAG dataset
downloaded from https://www.aminer.cn/oag-2-1. The original data have affiliation data
(`mag_affiliations.zip`), venue data (`mag_venues.zip`), author data (`mag_authors_*.zip`),
paper data (`mag_papers_*.zip`). All data are stored in JSON format.

To construct a graph for GNN training in GraphStorm, we need to preprocess the data
and convert them to the format supported by GraphStorm in
[here](https://github.com/awslabs/graphstorm/wiki/tutorials-own-data#use-own-data).
This data preprocessing is done in the Notebook (`MAG_parser.ipynb`) on a CPU machine.

After the data preprocessing, we also need to run `ComputeBERTEmbed.ipynb` to compute
BERT embeddings on the paper nodes and fos nodes. Please use a GPU machine to generate
BERT embeddings.

Once all the data are processed, run the following command to construct a graph
for distributed GNN training in GraphStorm. The command takes `mag_bert.json`
that specifies the input data for graph construction, constructs the graph,
splits it into 4 partitions and saves the partitions to `mag_bert_constructed3`.
The MAG graph is large, please use
r6a.48xlarge instance to construct the graph. Please prepare a disk to store
some intermediate data in the graph construction process. We can use
`--ext-mem-workspace` to specify where the intermediate data can be stored.

```
python3 -m graphstorm.gconstruct.construct_graph \
			--num-processes 16 \
			--output-dir mag_bert_constructed3 \
			--graph-name mag \
			--num-parts 4 \
			--skip-nonexist-edges \
			--ext-mem-workspace /mnt/raid0/tmp_oag \
			--ext-mem-feat-size 16 \
			--conf-file mag_bert.json
```

After constructing the graph, run the following command for distributed training.
```
python3 -m graphstorm.run.gs_link_prediction \
			--num-trainers 8 --num-servers 4 \
			--part-config mag_bert_constructed/mag.json \
			--ip-config ip_list.txt \
			--cf mag_lp.yaml \
			--num-epochs 1 \
			--save-model-path ~/mag_model \
			--node-feat-name fos:feat paper:feat 
```
