# GNN Distillation Tutorial

## 0. Pipeline Overview
GraphStorm supports to distill well-trained GNN models to user specified Transformer-based student model that can inference on any textual data, thus to 1) resolve the unseen node inference issue, and 2) remove the graph dependency for easy deployment . The distillation is conducted to minimize the embeddings between GNN checkpoint and student model. MSE is used to supervise the training.

To utilize the pipeline, user will need to input a textual dataset, where each sample corresponds to a GNN embeddings from teacher GNN model. The output of the pipeline will be the distilled Transformer-based student model that were trained to learn relational knowledge from GNN teacher model.


## 1. Required Input
### 1.1. Textual dataset

Textual dataset is required to be provided by user, which will be encoded by student Transformer-based model during distillation. User need to specify a path of directory with two sub-directory for ```train``` split and ```val``` split. In each split, there should be multiple partitions of ```*.parquet file```. See below example:
```
user_specified_data_root:
    train:
        0.parquet
        1.parquet
    val:
        0.parquet
        1.parquet
```
Each parquet file should be a DataFrame with two columns named by ```textual_feats``` and ```embeddings```. For each sample, ```textual_feats``` is the text data that represents a feature of a node, and ```embeddings``` is the GNN embeddings for the corresponding node. See example below for how the DataFrame is structured. “ids” column is optional. Note that “textual_feats” can be anything that user believes appropriate to represent a node.
```
ids        textual_feats                                     embeddings
B08RF12DJC item_name: Mimilure 150/600 Pcs Rubber Fishing... [-1.9489740133285522, 2.4145472049713135, -0.9...
B086DFT2BL item_name: Renzo's Kids Vitamin C with Elderbe... [-1.6588571071624756, -0.4345688223838806, -1....
B07HL6Q3C1 item_name: Costa Farms Chinese Evergreen Live ... [-2.2123806476593018, 1.666466236114502, -2.72...
B0041CZK0S item_name: Biofreeze Roll-On Pain-Relieving Ge... [1.6734728813171387, 0.2856823205947876, -0.66...
B07Q2CQBB4 item_name: SUNWILL 20oz Tumbler with Lid, Stai... [-0.5737705826759338, 0.8738870620727539, -0.2...
...  ... ...
B08GLL4MW5 item_name: FLYNOVA Rechargeable Hand Operated ... [-0.07194116711616516, -1.6652019023895264, 0....
B0BB9ZF1S3 item_name: GORGLITTER Men's Graphic Sweatshirt... [1.8190479278564453, 1.2688626050949097, -0.08...
B08JD3QBYG item_name: EyeLine Golf Groove Putting Mirror ... [0.7542105317115784, 1.1232998371124268, 0.281...
B09QZLHS6H item_name: CIMELR Kids Camera Toys for 6 7 8 9... [-0.6448838710784912, -0.7723445892333984, -0....
B06WXX64GL item_name: Alex Evenings Women's Long Rosette ... [1.8173906803131104, 1.6584017276763916, 0.105...
```
Textual dataset needs to be specified by GSF Yaml config under “distill” section. See example config below:
```
---
version: 1.0
gsf:
  ...
  distill:
    textual_data_path: <user_specified_data_root>
  ...

```
### 1.2. Saved Model Path

The second required input is the path to save the distilled student model. This needs to be specified in Yaml config under ```output``` section. See example below
```
---
version: 1.0
gsf:
  ...
  distill:
    textual_data_path: <user_specified_data_root>
  ...
  output:
    save_model_path: <user_specified_saved_model_path>
```
### 1.3. IP List

User needs to specify IP list of machines for distributed training. This list is a ```*.txt``` file, where each IP saved in a row. For example, an IP list with two instances file like the following:
```
10.2.14.2
10.2.76.33
```

User needs to specify the path of IP list in Yaml config:
```
---
version: 1.0
gsf:
  basic:
    backend: gloo # currently support gloo only
    ip_config: <user_specified_ip_list> # required input
  ...
```

## 2. Key Optional Inputs

### 2.1. Transformer-based LM model

In the case where only required inputs are specified, the student Transformer-based model would default to be [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) from HuggingFace, with pre-trained weights from “[distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)”. The distillation would be trained by default hyper-parameters described in Appendix A. 
 
Users can specify other LM model architectures by specifying the arguments:
```
---
version: 1.0
gsf:
  ...
  distill:
    ...
    lm_name: <user_specified_lm_model> # optional, default to be DistilBertModel
    ...
  ...
```

For user who wants to specify other hyper-parameters, please refer to Appendix A. for more optional inputs.


## 3. Output

The distilled Transformer-based student model would be the only output. The checkpoint will be saved in ```<save_model_path>/checkpoint-<global_step>``` specified by the user. For each checkpoint folder, two files will be saved: ```config.yaml``` to save the configs for distillation, and ```pytorch_model.bin``` to save the model weights.

## Running Command
```
python3 -m graphstorm.run.gs_gnn_distillation \
        --workspace /tmp/gsgnn_dt/ \
        --num-trainers 2 \
        --num-servers 2 \
        --num-samplers 0 \
        --ip-config /tmp/gsgnn_dt/ip_list.txt \
        --ssh-port 2222 \
        --cf /tmp/gsgnn_dt/config.yaml
```

## Appendix
### A. Default and optional Hyper-parameters
```
---
version: 1.0
gsf:
  basic:
    backend: gloo # currently support gloo only
    ip_config: <user_specified_ip_list> # required input
  distill:
    textual_data_path: <user_specified_data_root> # required input
    lm_name: DistilBertModel # optional, default to be DistilBertModel
    pretrained_weights: distilbert-base-uncased # optional, default to be null
  output:
    save_model_path: <user_specified_saved_model_path> # required input
    save_model_frequency: 1000 # optional, default to be 1000
  hyperparam:
    lm_tune_lr: 0.0001 # optional, default to be 0.0001
    num_epochs: 3 # optional, default to be 3
    batch_size: 128 # optional, default to be 128
    eval_frequency: 1000 # optional, default to be 1000
```