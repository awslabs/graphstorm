import json

def parse_bert_config(conf_path):
    with open(conf_path, "r") as f:
        config = json.load(f)

    bert_configs = {}
    for conf in config["bert_models"]:
        ntype = conf["node_type"]
        m_conf = conf["configs"]

        bert_configs[ntype] = m_conf

    return bert_configs