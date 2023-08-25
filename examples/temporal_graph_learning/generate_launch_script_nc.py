import yaml
import os
import sys

workspace = os.getcwd()
ip_config = "./ip_config.txt"
gs_config = "./graphstorm_train_script_nc_config.yaml"


train_data_path = os.path.join(workspace, 'DATA/MAG_Temporal/MAG_Temporal.json')
restore_model_path = None

save_model_path = os.path.join(workspace, 'model')
if restore_model_path != None:
    save_model_path += '_finetune'

yaml_dict = {
    "version": 1.0,
    "gsf": {
        "basic": {
            "backend": "gloo",
            "model_encoder_type": "rgcn",
            "verbose": False,
            "local_rank": 0,
        },
        "gnn": {
            "fanout": "20, 15",
            "num_layers": 2,
            "hidden_size": 16,
            "num_heads": 2,
            "use_mini_batch_infer": True,
        },
        "input": {"restore_model_path": restore_model_path},
        "output": {
            "save_model_path": save_model_path,
            "topk_model_to_save": 1,
        },
        "hyperparam": {
            "dropout": 0.1,
            "lr": 0.001,
            "sparse_optimizer_lr": 0.0001,
            "num_epochs": 1000,
            "batch_size": 512,
            "eval_batch_size": 512,
            "wd_l2norm": 1e-06,
            "save_model_frequency": 300,
            "eval_frequency": 300
        },
        "rgcn": {"num_bases": -1, "use_self_loop": True},
        "node_classification": {
            "node_feat_name": ["paper:feat"],
            "target_ntype": ["paper"],
            "eval_target_ntype": ["paper"],
            "label_field": "label",
            "multilabel": False,
            "num_classes": 152,
            "early_stop_rounds": 10,
            "early_stop_strategy": "consecutive_increase",
            "eval_metric": ["accuracy"],
        },
    },
    "udf": {
        "save_result_path": "tgat_nc_gpu",
    },
}

yaml.safe_dump(yaml_dict, open(gs_config, "w"))
with open(ip_config, "w") as f:
    f.write("127.0.0.1")

cmd = f"python3 -m graphstorm.run.launch \
            --workspace {workspace} \
            --part-config {train_data_path} \
            --ip-config {ip_config} \
            --num-trainers 1 \
            --num-servers 1 \
            --num-omp-threads 1 \
            --num-samplers 4 \
            --ssh-port 22 \
            main_nc.py --cf {gs_config}"

with open("run_script.sh", "w") as f:
    f.write(cmd + "\n")

print(">>> Run commnad", cmd)
