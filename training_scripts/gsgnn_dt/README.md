# Tutorial: Use GraphStorm for GNN Distillation

## 0. Pipeline Overview
GraphStorm supports to distill well-trained GNN models to a user specified Transformer-based student model that can inference on any textual data, thus to 1) resolve the unseen node inference issue, and 2) remove the graph dependency for easy deployment . The distillation is conducted to minimize the distance between the embeddings from a GNN checkpoint and the embeddings conducted by a student lm model. MSE loss is used to supervise the training.

To utilize the pipeline, user will need to input a textual dataset, where each sample corresponds to a GNN embeddings from teacher GNN model. The output of the pipeline will be the distilled Transformer-based student model that were trained to learn relational knowledge from GNN teacher model.


## 1. Required Input
Below we describe 4 required inputs that needs to be specified. For user who wants to know more about optional inputs and default hyper-parameters, please refer to section 3.1 for details.

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
ids  textual_feats                                     embeddings
B001 item_name: Mimilure 150/600 Pcs Rubber Fishing... [-1.9489740133285522, 2.4145472049713135, -0.9...
B002 item_name: Renzo's Kids Vitamin C with Elderbe... [-1.6588571071624756, -0.4345688223838806, -1....
B003 item_name: Costa Farms Chinese Evergreen Live ... [-2.2123806476593018, 1.666466236114502, -2.72...
B004 item_name: Biofreeze Roll-On Pain-Relieving Ge... [1.6734728813171387, 0.2856823205947876, -0.66...
B005 item_name: SUNWILL 20oz Tumbler with Lid, Stai... [-0.5737705826759338, 0.8738870620727539, -0.2...
...  ... ...
B096 item_name: FLYNOVA Rechargeable Hand Operated ... [-0.07194116711616516, -1.6652019023895264, 0....
B097 item_name: GORGLITTER Men's Graphic Sweatshirt... [1.8190479278564453, 1.2688626050949097, -0.08...
B098 item_name: EyeLine Golf Groove Putting Mirror ... [0.7542105317115784, 1.1232998371124268, 0.281...
B099 item_name: CIMELR Kids Camera Toys for 6 7 8 9... [-0.6448838710784912, -0.7723445892333984, -0....
B100 item_name: Alex Evenings Women's Long Rosette ... [1.8173906803131104, 1.6584017276763916, 0.105...
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
### 1.2. Transformer-based LM model
User needs to specify the LM model architecture and the pre-trained weights with the same naming tradition from HuggingFace. For example, user can specify the LM model architecture by the name of "[DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)", with pre-trained weights of "[distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)".

The model architecture and the pre-trained weights needs to be specified in the Yaml config under ```lm_models``` -> ```distill_lm_models section```. An example setup is shown below:
 
```
---
version: 1.0
lm_model:
  distill_lm_models:
    -
      lm_type: <user_specified_model_architecture> # e.g., DistilBertModel
      model_name: <user_specified_model_weights> # e.g., distilbert-base-uncased
gsf:
  ...
```
### 1.3. Saved Model Path

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
### 1.4. IP List

User needs to specify IP list of machines for distributed training. This list is a ```*.txt``` file, where each IP saved in a row. For example, an IP list with two instances file like the following:
```
10.0.0.1
10.0.0.2
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

## 2. Output

The distilled Transformer-based student model would be the only output. The checkpoint will be saved in ```<save_model_path>/checkpoint-<global_step>``` specified by the user. For each checkpoint folder, the tokenizer, the project layer and the LM model will be saved respectively. Find the structure below:
```
<save_model_path>/checkpoint-<global_step>
|- tokenizer
    special_tokens_map.json
    tokenizer_config.json
    tokenizer.json
    vocab.txt
|- lm
    config.json
    pytorch_model.bin
|- proj
    pytorch_model.bin
```

## 3. Running Command
### 3.1. An exemplary YAML file
```
---
version: 1.0
lm_model:
  distill_lm_models:
    -
      lm_type: <user_specified_model_architecture> # required input, e.g., DistilBertModel
      model_name: <user_specified_model_weights> # required input, e.g., distilbert-base-uncased
gsf:
  basic:
    backend: gloo # currently support gloo only
    ip_config: <user_specified_ip_list> # required input
  distill:
    textual_data_path: <user_specified_data_root> # required input
  output:
    save_model_path: <user_specified_saved_model_path> # required input
    save_model_frequency: 1000 # optional, default to be 1000
  hyperparam:
    lm_tune_lr: 0.0001 # optional, default to be 0.0001
    max_distill_steps: 10000 # optional, default to be 10000
    batch_size: 128 # optional, default to be 128
    eval_frequency: 1000 # optional, default to be 1000
```
### 3.2. Command
```
python3 -m graphstorm.run.gs_gnn_distillation \
        --workspace /tmp/gsgnn_dt/ \
        --num-trainers 2 \
        --num-samplers 0 \
        --ip-config /tmp/gsgnn_dt/ip_list.txt \
        --ssh-port 2222 \
        --cf /tmp/gsgnn_dt/config.yaml
```