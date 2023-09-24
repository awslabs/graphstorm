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

## Train GNN model to predict the venue of papers

### Construct the graph with venues as labels.
First, we need to construct the MAG graph with venues as the label of paper nodes.
```
python3 -m graphstorm.gconstruct.construct_graph \
			--num-processes 16 \
			--output-dir mag_4parts \
			--graph-name mag \
			--num-parts 4 \
			--skip-nonexist-edges \
			--ext-mem-workspace /mnt/raid0/tmp_oag \
			--ext-mem-feat-size 16 \
			--conf-file mag_v0.2.json \
			--add-reverse-edges
```
### Train GNN model with pre-trained BERT model

The command below runs pre-trained BERT model to compute BERT embeddings on
paper nodes and fos nodes and run RGCN to predict the venues.
```
python3 -m graphstorm.run.gs_node_classification \
            --num-trainers 8 \
            --num-servers 4 \
            --num-samplers 0 \
            --part-config mag_4parts/mag.json \
            --ip-config ip_list_4p.txt \
            --cf mag_gnn_nc.yaml
```

This method leads to the accuracy of 53.78%.

We can replace RGCN with HGT.

```
python3 -m graphstorm.run.gs_node_classification \
			--num-trainers 8 \
			--num-servers 4 \
			--num-samplers 0 \
			--part-config mag_4parts/mag.json \
			--ip-config ip_list_4p.txt \
			--cf mag_gnn_nc.yaml \
			--model-encoder-type hgt
```
On the full graph, HGT has much better performance than RGCN. The model reaches the accuracy of 59.12%.

### Fine-tune BERT model to predict the venue

We can fine-tune the BERT model on the paper nodes and predict the venue directly.
This can be done in GraphStorm with the following command. The trained model is saved
under `mag_bert_nc_model`.

```
python3 -m graphstorm.run.gs_node_classification \
		--num-trainers 8 \
		--num-servers 1 \
		--num-samplers 0 \
		--part-config mag_4parts/mag.json \
		--ip-config ip_list_4p.txt \
		--cf mag_bert_nc.yaml \
		--save-model-path mag_bert_nc_model
```

The accuracy is 41.88%.

### Fine-tune BERT model on the graph data and train GNN model to predict the venue

To achieve good performance, we should fine-tune the BERT model on the graph data.
One way of fine-tuning the BERT model on the graph data is to fine-tune the BERT model
with link prediction. This can be done in GraphStorm with the following command
and save the trained models under `mag_bert_lp_model`.

```
python3 -m graphstorm.run.gs_link_prediction \
			--num-trainers 8 \
			--num-servers 1 \
			--num-samplers 0 \
			--part-config mag_4parts/mag.json \
			--ip-config ip_list_4p.txt \
			--cf mag_bert_ft.yaml \
			--save-model-path mag_bert_lp_model
```

We can load the BERT model fine-tuned from link prediction to generate BERT embeddings
and train GNN model.

```
# train a RGCN model.
python3 -m graphstorm.run.gs_node_classification \
            --num-trainers 8 \
            --num-servers 4 \
            --num-samplers 0 \
            --part-config mag_4parts/mag.json \
            --ip-config ip_list_4p.txt \
            --cf mag_gnn_nc.yaml \
            --restore-model-path mag_bert_lp_model/epoch-2 \
            --restore-model-layers dense_embed

# train a HGT model.
python3 -m graphstorm.run.gs_node_classification \
            --num-trainers 8 \
            --num-servers 4 \
            --num-samplers 0 \
            --part-config mag_4parts/mag.json \
            --ip-config ip_list_4p.txt \
            --cf mag_gnn_nc.yaml \
            --restore-model-path mag_bert_lp_model/epoch-2 \
            --restore-model-layers dense_embed \
            --model-encoder-type hgt
```

The accuracy of RGCN with the BERT model fine-tuned with link prediction is 57.86%,
while the accuracy of HGT is 62.09%.

We can also train a GNN model with the BERT model fine-tuned for predicting venues.

```
python3 -m graphstorm.run.gs_node_classification \
            --num-trainers 8 \
            --num-servers 4 \
            --num-samplers 0 \
            --part-config mag_4parts/mag.json \
            --ip-config ip_list_4p.txt \
            --cf mag_gnn_nc.yaml \
            --restore-model-path mag_bert_nc_model/epoch-9/ \
            --restore-model-layers dense_embed \
			--save-model-path mag_rgcn_model

python3 -m graphstorm.run.gs_node_classification \
			--num-trainers 8 \
			--num-servers 4 \
			--num-samplers 0 \
			--part-config mag_4parts/mag.json \
			--ip-config ip_list_4p.txt \
			--cf mag_gnn_nc.yaml \
			--restore-model-path mag_bert_nc_model/epoch-6/ \
			--restore-model-layers dense_embed \
			--model-encoder-type hgt \
			--save-model-path mag_hgt_model
```

The accuracy of RGCN with the BERT model fine-tuned with venue prediction is 63.22%,
while the accuracy of HGT is 67.20%.
