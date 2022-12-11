import os, sys
from pathlib import Path
from tempfile import tempdir
import yaml
import unittest, pytest
from argparse import Namespace

from graphstorm.config import GSConfig
from graphstorm.config.config import BUILTIN_LP_LOSS_CROSS_ENTROPY
from graphstorm.config.config import BUILTIN_LP_LOSS_LOGSIGMOID_RANKING
from graphstorm.dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER
from graphstorm.config.config import GRAPHSTORM_SAGEMAKER_TASK_TRACKER

def check_failure(config, field):
    has_error = False
    try:
        dummy = getattr(config, field)
    except:
        has_error = True
    assert has_error

def create_dummpy_config_obj():
    yaml_object = { # dummy config, bypass checks by default
        "version": 1.0,
        "gsf": {
            "basic": {},
            "gnn": {
                "fanout": "4",
                "n_layers": 1,
            },
            "input": {},
            "output": {},
            "hyperparam": {
                "lr": 0.01,
                "sparse_lr": 0.0001
            },
            "rgcn": {},
        }
    }
    return yaml_object

def create_basic_config(tmp_path, file_name):
    yaml_object = create_dummpy_config_obj()
    yaml_object["gsf"]["basic"] = {
        "debug" : True,
        "graph_name": "test",
        "backend": "gloo",
        "num_gpus": 1,
        "ip_config": "ip.txt",
        "part_config": "part.json",
        "model_encoder_type": "rgat",
        "evaluation_frequency": 100,
        "no_validation": True,
        "mixed_precision": True, # TODO(xiangsx) TMP: will Fail
    }

    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # config for check default value
    yaml_object["gsf"]["basic"] = {
        "num_gpus": 1,
        "ip_config": "ip.txt",
        "part_config": "part.json",
    }

    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # config for wrong values
    yaml_object["gsf"]["basic"] = {
        "backend": "error",
        "num_gpus": 0,
        "evaluation_frequency": 0,
        "model_encoder_type": "abc"
    }

    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_load_basic_info():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_basic_config(Path(tmpdirname), 'basic_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'basic_test.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        # success load
        assert config.debug == True
        assert config.graph_name == "test"
        assert config.backend == "gloo"
        assert config.num_gpus == 1
        assert config.ip_config == "ip.txt"
        assert config.part_config == "part.json"
        assert config.verbose == False
        assert config.evaluation_frequency == 100
        assert config.no_validation == True
        check_failure(config, "mixed_precision")

        # Change config's variables to do further testing
        config._backend = "nccl"
        assert config.backend == "nccl"
        config._model_encoder_type = "lm"
        assert config.model_encoder_type == "lm"

        # Check default values
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'basic_test_default.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.debug == False
        check_failure(config, "graph_name")
        assert config.backend == "gloo"
        assert config.evaluation_frequency == sys.maxsize
        assert config.no_validation == False
        assert config.mixed_precision == False
        check_failure(config, "model_encoder_type") # must provide model_encoder_type

        # Check failures
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'basic_test_fail.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        check_failure(config, "backend")
        check_failure(config, "num_gpus")
        check_failure(config, "ip_config")
        check_failure(config, "part_config")
        check_failure(config, "evaluation_frequency")
        check_failure(config, "model_encoder_type")

def create_gnn_config(tmp_path, file_name):
    yaml_object = create_dummpy_config_obj()
    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "rgat"
    }
    yaml_object["gsf"]["gnn"] = {
        "feat_name": "test_feat",
        "fanout": "10,20,30",
        "n_layers": 3,
        "n_hidden": 128,
        "mini_batch_infer": False
    }
    with open(os.path.join(tmp_path, file_name+"1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "rgcn"
    }
    yaml_object["gsf"]["gnn"] = {
        "feat_name": "test_feat",
        "fanout": "a:10@b:10,a:10@b:10@c:20",
        "eval_fanout": "10,10",
        "n_layers": 2,
        "n_hidden": 128,
        "mini_batch_infer": True
    }
    with open(os.path.join(tmp_path, file_name+"2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "lm"
    }
    yaml_object["gsf"]["gnn"] = {
        "n_layers": 2, # for encoder of lm, n_layers will always be 0
        "n_hidden": 128,
    }
    with open(os.path.join(tmp_path, file_name+"3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # config for check default value
    yaml_object["gsf"]["gnn"] = {
    }

    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "rgcn"
    }
    yaml_object["gsf"]["gnn"] = {
        "fanout": "error", # error fanout
        "eval_fanout": "error",
        "n_hidden": 0,
        "n_layers": 0,
        "mini_batch_infer": "error"
    }
    with open(os.path.join(tmp_path, file_name+"_error1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["gnn"] = {
        "fanout": "10,10", # error fanout
        "eval_fanout": "10,10",
        "n_hidden": 32,
        "n_layers": 1,
    }
    with open(os.path.join(tmp_path, file_name+"_error2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)


def test_gnn_info():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_gnn_config(Path(tmpdirname), 'gnn_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test1.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.feat_name == "test_feat"
        assert config.fanout == [10,20,30]
        assert config.eval_fanout == [-1, -1, -1]
        assert config.n_layers == 3
        assert config.n_hidden == 128
        assert config.mini_batch_infer == False

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test2.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.feat_name == "test_feat"
        assert config.fanout[0]["a"] == 10
        assert config.fanout[0]["b"] == 10
        assert config.fanout[1]["a"] == 10
        assert config.fanout[1]["b"] == 10
        assert config.fanout[1]["c"] == 20
        assert config.eval_fanout == [10,10]
        assert config.n_layers == 2
        assert config.n_hidden == 128
        assert config.mini_batch_infer == True

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test3.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.n_layers == 0 # lm model does not need n layers

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test_default.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.feat_name is None
        assert config.n_layers == 0 # lm model does not need n layers
        assert config.n_hidden == 0 # lm model may not need n hidden
        assert config.mini_batch_infer == True
        check_failure(config, "fanout") # fanout must be provided if used
        check_failure(config, "eval_fanout")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test_error1.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        check_failure(config, "fanout")
        check_failure(config, "eval_fanout")
        check_failure(config, "n_hidden")
        check_failure(config, "n_layers")
        check_failure(config, "mini_batch_infer")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test_error2.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        check_failure(config, "fanout")
        check_failure(config, "eval_fanout")

def create_io_config(tmp_path, file_name):
    yaml_object = create_dummpy_config_obj()
    yaml_object["gsf"]["input"] = {
    }
    yaml_object["gsf"]["output"] = {
    }

    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["input"] = {
        "restore_model_path": "./restore",
        "restore_optimizer_path": "./opt_restore",
    }

    yaml_object["gsf"]["output"] = {
        "save_model_path": os.path.join(tmp_path, "save"),
        "save_model_per_iters": 100,
        "save_embeds_path": "./save_emb",
    }

    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_load_io_info():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_io_config(Path(tmpdirname), 'io_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'io_test_default.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.restore_model_path == None
        assert config.restore_optimizer_path == None
        assert config.save_model_path == None
        assert config.save_model_per_iters == -1
        assert config.save_embeds_path == None

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'io_test.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.restore_model_path == "./restore"
        assert config.restore_optimizer_path == "./opt_restore"
        assert config.save_model_path == os.path.join(Path(tmpdirname), "save")
        assert config.save_model_per_iters == 100
        assert config.save_embeds_path == "./save_emb"

def create_task_tracker_config(tmp_path, file_name):
    yaml_object = create_dummpy_config_obj()
    yaml_object["gsf"]["output"] = {
    }

    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["output"] = {
        "task_tracker": "sagemaker_task_tracker",
        "log_report_frequency": 100,
    }

    # config for check default value
    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["output"] = {
        "task_tracker": "mlflow",
        "log_report_frequency": 0,
    }

    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_task_tracker_info():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_task_tracker_config(Path(tmpdirname), 'task_tracker_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'task_tracker_test_default.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.task_tracker == GRAPHSTORM_SAGEMAKER_TASK_TRACKER
        assert config.log_report_frequency == 1000

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'task_tracker_test.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.task_tracker == GRAPHSTORM_SAGEMAKER_TASK_TRACKER
        assert config.log_report_frequency == 100

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'task_tracker_test_fail.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        check_failure(config, "task_tracker")
        check_failure(config, "log_report_frequency")

def create_train_config(tmp_path, file_name):
    yaml_object = create_dummpy_config_obj()
    yaml_object["gsf"]["hyperparam"] = {
    }

    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # normal value
    yaml_object["gsf"]["hyperparam"] = {
        "dropout" : 0.1,
        "lr": 0.001,
        "n_epochs": 10,
        "batch_size": 64,
        "eval_batch_size": 128,
        "wd_l2norm": 0.1,
        "alpha_l2norm": 0.00001,
        "topk_model_to_save": 3,
        "sparse_lr": 0.001,
        "use_node_embeddings": False,
        "use_self_loop": False,
        "enable_early_stop": True,
        "save_model_path": os.path.join(tmp_path, "save"),
    }

    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # for failures
    yaml_object["gsf"]["hyperparam"] = {
        "dropout" : -1.0,
        "lr": 0.,
        "n_epochs": -1,
        "batch_size": 0,
        "eval_batch_size": 0,
        "sparse_lr": 0.,
        "use_node_embeddings": True,
        "use_self_loop": "error",
        "enable_early_stop": True,
        "call_to_consider_early_stop": -1,
        "window_for_early_stop": 0,
    }

    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)


def test_train_info():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_train_config(Path(tmpdirname), 'train_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'train_test_default.yaml'), local_rank=0)
        config = GSConfig(args)

        assert config.dropout == 0
        check_failure(config, "lr")
        assert config.n_epochs == 0
        check_failure(config, "batch_size")
        config._batch_size = 32
        assert config.batch_size == 32
        assert config.eval_batch_size == 32
        assert config.wd_l2norm == 0
        assert config.alpha_l2norm == 0
        assert config.topk_model_to_save == 0
        config._lr = 0.01
        assert config.sparse_lr == 0.01
        assert config.use_node_embeddings == False
        assert config.use_self_loop == True
        assert config.enable_early_stop == False

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'train_test.yaml'), local_rank=0)
        config = GSConfig(args)

        assert config.dropout == 0.1
        assert config.lr == 0.001
        assert config.n_epochs == 10
        assert config.batch_size == 64
        assert config.eval_batch_size == 128
        assert config.wd_l2norm == 0.1
        assert config.alpha_l2norm == 0.00001
        assert config.topk_model_to_save == 3
        assert config.sparse_lr == 0.001
        assert config.use_node_embeddings == False
        assert config.use_self_loop == False
        assert config.enable_early_stop == True
        assert config.call_to_consider_early_stop == 0
        assert config.window_for_early_stop == 3

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'train_test_fail.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "dropout")
        check_failure(config, "lr")
        check_failure(config, "n_epochs")
        check_failure(config, "batch_size")
        check_failure(config, "eval_batch_size")
        check_failure(config, "sparse_lr")
        assert config.use_node_embeddings == True
        check_failure(config, "use_self_loop")
        config._dropout = 1.0
        check_failure(config, "dropout")
        assert config.enable_early_stop == True
        check_failure(config, "call_to_consider_early_stop")
        check_failure(config, "window_for_early_stop")

def create_rgcn_config(tmp_path, file_name):
    yaml_object = create_dummpy_config_obj()
    yaml_object["gsf"]["rgcn"] = {
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["rgcn"] = {
        "n_bases": 2,
    }
    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["rgcn"] = {
        "n_bases": 0.1,
    }
    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["rgcn"] = {
        "n_bases": -2,
    }
    with open(os.path.join(tmp_path, file_name+"_fail2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)



def test_rgcn_info():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_rgcn_config(Path(tmpdirname), 'rgcn_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'rgcn_test_default.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.n_bases == -1

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'rgcn_test.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.n_bases == 2

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'rgcn_test_fail.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "n_bases")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'rgcn_test_fail2.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "n_bases")

def create_rgat_config(tmp_path, file_name):
    yaml_object = create_dummpy_config_obj()
    yaml_object["gsf"]["rgat"] = {
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["rgat"] = {
        "n_heads": 2,
    }
    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["rgat"] = {
        "n_heads": 0,
    }
    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_rgat_info():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_rgat_config(Path(tmpdirname), 'rgat_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'rgat_test_default.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.n_heads == 4

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'rgat_test.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.n_heads == 2

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'rgat_test_fail.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "n_heads")

def create_node_class_config(tmp_path, file_name):
    yaml_object = create_dummpy_config_obj()
    yaml_object["gsf"]["node_classification"] = {
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_classification"] = {
        "predict_ntype": "a",
        "label_field": "label",
        "multilabel": True,
        "num_classes": 20,
    }
    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_classification"] = {
        "predict_ntype": "a",
        "label_field": "label",
        "multilabel": True,
        "imbalance_class_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2",
        "num_classes": 20,
    }
    with open(os.path.join(tmp_path, file_name+"1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval metric
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "eval_metric": "F1_score",
        "imbalance_class_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2",
    }
    with open(os.path.join(tmp_path, file_name+"_metric1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval metric
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "eval_metric": ["F1_score", "precision_recall", "ROC_AUC"],
        "imbalance_class_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2",
    }
    with open(os.path.join(tmp_path, file_name+"_metric2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_classification"] = {
        "multilabel": "error",
        "num_classes": 0,
    }
    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval metric and multi-label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "multilabel_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2", # multilabel is not set to True
        "eval_metric": "unknown"
    }

    with open(os.path.join(tmp_path, file_name+"_fail_metric1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval metric and multi-label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "eval_metric": {}, # eval metric must be string or list
        "multilabel": False,
        "multilabel_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2", # Do not need multilabel_weights
    }

    with open(os.path.join(tmp_path, file_name+"_fail_ml_w1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval metric and multi-label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "eval_metric": ["F1_score", "unknown"], # one of metrics is not supported
        "multilabel": True,
        "multilabel_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,0.1,0.2", # Does not match num_classes
    }

    with open(os.path.join(tmp_path, file_name+"_fail_ml_w2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test multi-label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "multilabel": True,
        "multilabel_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,0.1,0.2,0.3,0.1,-0.1", # weight can not be negative
    }

    with open(os.path.join(tmp_path, file_name+"_fail_ml_w3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test imbalance label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "imbalance_class_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,0.1,0.2,0.3,0.1", # len(weight) != num_classes
    }

    with open(os.path.join(tmp_path, file_name+"_fail_imb_l_w1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test imbalance label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "imbalance_class_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,0.1,0.2,0.3,0.1,0", # weight must larger than 0
    }

    with open(os.path.join(tmp_path, file_name+"_fail_imb_l_w2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test imbalance label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "multilabel": True,
        "imbalance_class_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,0.1,0.2,0.3,0.1,0", # Does not work with multilabel
    }

    with open(os.path.join(tmp_path, file_name+"_fail_imb_l_w3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test imbalance label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "imbalance_class_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,0.1,0.2,0.3,0.1,abc123", # weights must be float
    }

    with open(os.path.join(tmp_path, file_name+"_fail_imb_l_w4.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_node_class_info():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_node_class_config(Path(tmpdirname), 'node_class_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_default.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "predict_ntype")
        check_failure(config, "label_field")
        assert config.multilabel == False
        assert config.multilabel_weights == None
        assert config.imbalance_class_weights == None
        check_failure(config, "num_classes")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.predict_ntype == "a"
        assert config.label_field == "label"
        assert config.multilabel == True
        assert config.multilabel_weights == None
        assert config.imbalance_class_weights == None
        assert config.num_classes == 20
        assert len(config.eval_metric) == 1
        assert config.eval_metric[0] == "accuracy"

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.multilabel == True
        # imbalance_class_weight does not work with multilabel == True
        check_failure(config, "imbalance_class_weights")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_metric1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.eval_metric[0] == "f1_score"
        assert config.imbalance_class_weights.tolist() == [1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2]

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_metric2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert len(config.eval_metric) == 3
        assert config.eval_metric[0] == "f1_score"
        assert config.eval_metric[1] == "precision_recall"
        assert config.eval_metric[2] == "roc_auc"

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "multilabel")
        check_failure(config, "num_classes")
        check_failure(config, "eval_metric")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_metric1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "eval_metric")
        check_failure(config, "multilabel_weights")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_ml_w1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "eval_metric")
        assert config.multilabel == False
        check_failure(config, "multilabel_weights")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_ml_w2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "eval_metric")
        assert config.multilabel == True
        check_failure(config, "multilabel_weights")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_ml_w3.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        assert config.multilabel == True
        check_failure(config, "multilabel_weights")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_imb_l_w1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "imbalance_class_weights")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_imb_l_w2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "imbalance_class_weights")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_imb_l_w3.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        assert config.multilabel == True
        check_failure(config, "imbalance_class_weights")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_imb_l_w4.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "imbalance_class_weights")

def create_node_regress_config(tmp_path, file_name):
    yaml_object = create_dummpy_config_obj()
    yaml_object["gsf"]["node_regression"] = {
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_regression"] = {
        "predict_ntype": "a",
        "label_field": "label",
        "eval_metric": "Mse"
    }
    with open(os.path.join(tmp_path, file_name+"1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_regression"] = {
        "predict_ntype": "a",
        "label_field": "label",
        "eval_metric": ["mse", "RMSE"],
    }
    with open(os.path.join(tmp_path, file_name+"2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_regression"] = {
        "eval_metric": "error"
    }
    with open(os.path.join(tmp_path, file_name+"_fail_metric1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_regression"] = {
        "eval_metric": ["MSE", "error"], # one of metrics is not supported
    }
    with open(os.path.join(tmp_path, file_name+"_fail_metric2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_regression"] = {
        "eval_metric": {}, # eval metric must be string or list
    }
    with open(os.path.join(tmp_path, file_name+"_fail_metric3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_node_regress_info():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_node_regress_config(Path(tmpdirname), 'node_regress_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_regress_test_default.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "predict_ntype")
        check_failure(config, "label_field")
        assert len(config.eval_metric) == 1
        assert config.eval_metric[0] == "rmse"

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_regress_test1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.predict_ntype == "a"
        assert config.label_field == "label"
        assert len(config.eval_metric) == 1
        assert config.eval_metric[0] == "mse"

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_regress_test2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert len(config.eval_metric) == 2
        assert config.eval_metric[0] == "mse"
        assert config.eval_metric[1] == "rmse"

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_regress_test_fail_metric1.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "eval_metric")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_regress_test_fail_metric2.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "eval_metric")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_regress_test_fail_metric3.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "eval_metric")

def create_edge_class_config(tmp_path, file_name):
    yaml_object = create_dummpy_config_obj()
    yaml_object["gsf"]["edge_classification"] = {
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["edge_classification"] = {
        "target_etype": ["query,match,asin"],
        "reverse_edge_types_map": [],
        "label_field": "label",
        "multilabel": True,
        "num_classes": 4,
        "num_decoder_basis": 4,
        "remove_target_edge": False,
        "decoder_type": "MLPDecoder"
    }

    with open(os.path.join(tmp_path, file_name+"1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["edge_classification"] = {
        "target_etype": ["query,match,asin", "query,click,asin"],
        "reverse_edge_types_map": ["query,match,rev-match,asin", "query,click,rev-click,asin"],
        "num_classes": 4,
        "eval_metric": ["Per_class_f1_score", "Precision_Recall"]
    }

    with open(os.path.join(tmp_path, file_name+"2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test failure
    yaml_object["gsf"]["edge_classification"] = {
        "target_etype": "query,match,asin",
        "reverse_edge_types_map": "query,match,rev-match,asin",
        "multilabel": "error",
        "num_classes": 1,
        "num_decoder_basis": 1,
        "remove_target_edge": "error",
    }

    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["edge_classification"] = {
        "target_etype": [],
        "num_classes": 4,
        "eval_metric": ["per_class_f1_score", "rmse"]
    }
    with open(os.path.join(tmp_path, file_name+"_fail2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_edge_class_info():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_edge_class_config(Path(tmpdirname), 'edge_class_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'edge_class_test_default.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "target_etype")
        assert config.decoder_type == "DenseBiDecoder"
        assert config.num_decoder_basis == 2
        assert config.remove_target_edge == True
        assert len(config.reverse_edge_types_map) == 0
        check_failure(config, "label_field")
        assert config.multilabel == False
        check_failure(config, "num_classes")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'edge_class_test1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.target_etype[0] == ("query", "match", "asin")
        assert len(config.target_etype) == 1
        assert config.decoder_type == "MLPDecoder"
        assert config.num_decoder_basis == 4
        assert config.remove_target_edge == False
        assert len(config.reverse_edge_types_map) == 0
        assert config.label_field == "label"
        assert config.multilabel == True
        assert config.num_classes == 4
        assert len(config.eval_metric) == 1
        assert config.eval_metric[0] == "accuracy"

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'edge_class_test2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.target_etype[0] == ("query", "match", "asin")
        assert config.target_etype[1] == ("query", "click", "asin")
        assert len(config.target_etype) == 2
        assert len(config.reverse_edge_types_map) == 2
        print(config.reverse_edge_types_map)
        assert config.reverse_edge_types_map[("query","match","asin")] == \
             ("asin","rev-match","query")
        assert config.reverse_edge_types_map[("query","click","asin")] == \
             ("asin","rev-click","query")
        assert len(config.eval_metric) == 2
        assert config.eval_metric[0] == "per_class_f1_score"
        assert config.eval_metric[1] == "precision_recall"

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'edge_class_test_fail.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "target_etype")
        check_failure(config, "reverse_edge_types_map")
        check_failure(config, "multilabel")
        check_failure(config, "num_classes")
        check_failure(config, "num_decoder_basis")
        check_failure(config, "remove_target_edge")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'edge_class_test_fail2.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "target_etype")
        check_failure(config, "eval_metric")

def create_lp_config(tmp_path, file_name):
    yaml_object = create_dummpy_config_obj()
    yaml_object["gsf"]["link_prediction"] = {
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "negative_sampler": BUILTIN_LP_JOINT_NEG_SAMPLER,
        "num_negative_edges": 4,
        "num_negative_edges_eval": 100,
        "train_etype": ["query,exactmatch,asin"],
        "eval_etype": ["query,exactmatch,asin"],
        "separate_eval": True,
        "exclude_training_targets": True,
        "reverse_edge_types_map": ["query,exactmatch,rev-exactmatch,asin"],
        "gamma": 2.0,
        "lp_loss_func": BUILTIN_LP_LOSS_LOGSIGMOID_RANKING,
        "eval_metric": "MRR",
        "use_dot_product": True,
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "negative_sampler": "udf", # we allow udf sampler
        "train_etype": ["query,exactmatch,asin","query,click,asin"],
        "eval_etype": ["query,exactmatch,asin","query,click,asin"],
        "separate_eval": True,
        "exclude_training_targets": False,
        "reverse_edge_types_map": None,
        "eval_metric": ["mrr"],
        "gamma": 1.0,
    }
    with open(os.path.join(tmp_path, file_name+"2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "num_negative_edges": 0,
        "num_negative_edges_eval": 0,
        "train_etype": "query,exactmatch,asin",
        "eval_etype": "query,exactmatch,asin",
        "separate_eval": "error",
        "exclude_training_targets": "error",
        "reverse_edge_types_map": "query,exactmatch,rev-exactmatch,asin",
        "lp_loss_func": "unknown",
        "use_dot_product": "false",
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_fail1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "exclude_training_targets": True,
        "reverse_edge_types_map": [],
    }
    with open(os.path.join(tmp_path, file_name+"_fail2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "eval_metric": "error"
    }
    with open(os.path.join(tmp_path, file_name+"_fail_metric1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "eval_metric": ["mrr", "error"], # one of metrics is not supported
    }
    with open(os.path.join(tmp_path, file_name+"_fail_metric2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "eval_metric": {}, # eval metric must be string or list
    }
    with open(os.path.join(tmp_path, file_name+"_fail_metric3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_lp_info():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_lp_config(Path(tmpdirname), 'lp_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_default.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.negative_sampler == BUILTIN_LP_UNIFORM_NEG_SAMPLER
        assert config.num_negative_edges == 16
        assert config.num_negative_edges_eval == 1000
        assert config.use_dot_product == False
        assert config.train_etype == None
        assert config.eval_etype == None
        assert config.separate_eval == False
        check_failure(config, "exclude_training_targets")
        assert len(config.reverse_edge_types_map) == 0
        assert config.gamma == 12.0
        assert config.lp_loss_func == BUILTIN_LP_LOSS_CROSS_ENTROPY
        assert len(config.eval_metric) == 1
        assert config.eval_metric[0] == "mrr"
        assert config.gamma == 12.0

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.negative_sampler == BUILTIN_LP_JOINT_NEG_SAMPLER
        assert config.num_negative_edges == 4
        assert config.num_negative_edges_eval == 100
        assert config.use_dot_product == True
        assert len(config.train_etype) == 1
        assert config.train_etype[0] == ("query", "exactmatch", "asin")
        assert len(config.eval_etype) == 1
        assert config.eval_etype[0] == ("query", "exactmatch", "asin")
        assert config.separate_eval == True
        assert config.exclude_training_targets == True
        assert len(config.reverse_edge_types_map) == 1
        assert config.reverse_edge_types_map[("query", "exactmatch","asin")] == \
            ("asin", "rev-exactmatch","query")
        assert config.gamma == 2.0
        assert config.lp_loss_func == BUILTIN_LP_LOSS_LOGSIGMOID_RANKING
        assert len(config.eval_metric) == 1
        assert config.eval_metric[0] == "mrr"

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.negative_sampler == "udf"
        assert len(config.train_etype) == 2
        assert config.train_etype[0] == ("query", "exactmatch", "asin")
        assert config.train_etype[1] == ("query", "click", "asin")
        assert len(config.eval_etype) == 2
        assert config.eval_etype[0] == ("query", "exactmatch", "asin")
        assert config.eval_etype[1] == ("query", "click", "asin")
        assert config.exclude_training_targets == False
        assert len(config.reverse_edge_types_map) == 0
        assert len(config.eval_metric) == 1
        assert config.eval_metric[0] == "mrr"
        assert config.gamma == 1.0

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_fail1.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "num_negative_edges")
        check_failure(config, "num_negative_edges_eval")
        check_failure(config, "train_etype")
        check_failure(config, "eval_etype")
        check_failure(config, "separate_eval")
        check_failure(config, "exclude_training_targets")
        check_failure(config, "reverse_edge_types_map")
        check_failure(config, "lp_loss_func")
        check_failure(config, "use_dot_product")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_fail2.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "exclude_training_targets")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_fail_metric1.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "eval_metric")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_fail_metric2.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "eval_metric")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_fail_metric3.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "eval_metric")

def create_lml_config(tmp_path, file_name):
    yaml_object = create_dummpy_config_obj()
    yaml_object["gsf"]["mlm"] = {
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["mlm"] = {
        "mlm_probability": 0.4
    }
    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["mlm"] = {
        "mlm_probability": 0.0
    }
    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_lml_info():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_lml_config(Path(tmpdirname), 'lml_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lml_test_default.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.mlm_probability == 0.15
        assert config.eval_metric == None

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lml_test.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.mlm_probability == 0.4

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lml_test_fail.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "mlm_probability")

def create_gnn_config(tmp_path, file_name):
    yaml_object = create_dummpy_config_obj()
    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "rgat"
    }
    yaml_object["gsf"]["gnn"] = {
        "feat_name": "test_feat",
        "fanout": "10,20,30",
        "n_layers": 3,
        "n_hidden": 128,
        "mini_batch_infer": False
    }
    with open(os.path.join(tmp_path, file_name+"1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "rgcn"
    }
    yaml_object["gsf"]["gnn"] = {
        "feat_name": "ntype0:feat_name",
        "fanout": "a:10@b:10,a:10@b:10@c:20",
        "eval_fanout": "10,10",
        "n_layers": 2,
        "n_hidden": 128,
        "mini_batch_infer": True
    }
    with open(os.path.join(tmp_path, file_name+"2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["gnn"] = {
        "feat_name": "ntype0:feat_name ntype1:fname",
    }
    with open(os.path.join(tmp_path, file_name+"3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "lm"
    }
    yaml_object["gsf"]["gnn"] = {
        "n_layers": 2, # for encoder of lm, n_layers will always be 0
        "n_hidden": 128,
    }
    with open(os.path.join(tmp_path, file_name+"4.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # config for check default value
    yaml_object["gsf"]["gnn"] = {
    }

    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "rgcn"
    }
    yaml_object["gsf"]["gnn"] = {
        "feat_name": "ntype0:feat_name ntype0:feat_name", # set feat_name twice
        "fanout": "error", # error fanout
        "eval_fanout": "error",
        "n_hidden": 0,
        "n_layers": 0,
        "mini_batch_infer": "error"
    }
    with open(os.path.join(tmp_path, file_name+"_error1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["gnn"] = {
        "feat_name": {"ntype0":"feat_name"}, # not a string
        "fanout": "10,10", # error fanout
        "eval_fanout": "10,10",
        "n_hidden": 32,
        "n_layers": 1,
    }
    with open(os.path.join(tmp_path, file_name+"_error2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)


def test_gnn_info():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_gnn_config(Path(tmpdirname), 'gnn_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test1.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.feat_name == "test_feat"
        assert config.fanout == [10,20,30]
        assert config.eval_fanout == [-1, -1, -1]
        assert config.n_layers == 3
        assert config.n_hidden == 128
        assert config.mini_batch_infer == False

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test2.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert len(config.feat_name) == 1
        assert 'ntype0' in config.feat_name
        assert config.feat_name['ntype0'] == "feat_name"
        assert config.fanout[0]["a"] == 10
        assert config.fanout[0]["b"] == 10
        assert config.fanout[1]["a"] == 10
        assert config.fanout[1]["b"] == 10
        assert config.fanout[1]["c"] == 20
        assert config.eval_fanout == [10,10]
        assert config.n_layers == 2
        assert config.n_hidden == 128
        assert config.mini_batch_infer == True

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test3.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert len(config.feat_name) == 2
        assert 'ntype0' in config.feat_name
        assert 'ntype1' in config.feat_name
        assert config.feat_name['ntype0'] == "feat_name"
        assert config.feat_name['ntype1'] == "fname"

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test4.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.n_layers == 0 # lm model does not need n layers

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test_default.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.feat_name is None
        assert config.n_layers == 0 # lm model does not need n layers
        check_failure(config, "n_hidden") # lm model may not need n hidden
        assert config.mini_batch_infer == True
        check_failure(config, "fanout") # fanout must be provided if used
        check_failure(config, "eval_fanout")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test_error1.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        check_failure(config, "feat_name")
        check_failure(config, "fanout")
        check_failure(config, "eval_fanout")
        check_failure(config, "n_hidden")
        check_failure(config, "n_layers")
        check_failure(config, "mini_batch_infer")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test_error2.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        check_failure(config, "feat_name")
        check_failure(config, "fanout")
        check_failure(config, "eval_fanout")

def create_io_config(tmp_path, file_name):
    yaml_object = create_dummpy_config_obj()
    yaml_object["gsf"]["input"] = {
    }
    yaml_object["gsf"]["output"] = {
    }

    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["input"] = {
        "restore_model_path": "./restore",
        "restore_optimizer_path": "./opt_restore",
    }

    yaml_object["gsf"]["output"] = {
        "save_model_path": os.path.join(tmp_path, "save"),
        "save_model_per_iters": 100,
        "save_embeds_path": "./save_emb",
    }

    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["output"] = {
        "save_model_path": os.path.join(tmp_path, "save"),
        "save_model_per_iters": 100,
        "save_embeds_path": "./save_emb",
        "save_predict_path": "./prediction",
    }

    with open(os.path.join(tmp_path, file_name+"2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_load_io_info():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_io_config(Path(tmpdirname), 'io_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'io_test_default.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.restore_model_path == None
        assert config.restore_optimizer_path == None
        assert config.save_model_path == None
        assert config.save_model_per_iters == -1
        assert config.save_embeds_path == None

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'io_test.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.restore_model_path == "./restore"
        assert config.restore_optimizer_path == "./opt_restore"
        assert config.save_model_path == os.path.join(Path(tmpdirname), "save")
        assert config.save_model_per_iters == 100
        assert config.save_embeds_path == "./save_emb"
        assert config.save_predict_path == "./save_emb"

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'io_test2.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.save_embeds_path == "./save_emb"
        assert config.save_predict_path == "./prediction"

if __name__ == '__main__':
    test_load_basic_info()
    test_gnn_info()
    test_load_io_info()
    test_train_info()
    test_rgcn_info()
    test_rgat_info()
    test_node_class_info()
    test_node_regress_info()
    test_edge_class_info()
    test_lp_info()
    test_lml_info()
