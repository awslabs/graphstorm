"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    GSgnn HAT masked language model pre-training
"""

import os
import torch as th
from dataclasses import dataclass, field
from typing import Optional

from transformer import (HfArgumentParser,
                         TrainingArguments)

import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig

from graphstorm.dataloading import GSgnnNodeTrainData
from graphstorm.trainer import GSgnnHATMasedLMTrainer
from graphstorm.utils import rt_profiler, sys_tracker, setup_device

from graphstorm.dataloading import GSlmHatNodeDataLoader
from graphstorm.model.graph_transformer import prepare_hat_node_centric, BFS_TRANSVERSE


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    max_sentence_length: int = field(
        default=32,
        metadata={
            "help": "The maximum number of sentences after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )

    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    transverse_format: str = field(
        default=BFS_TRANSVERSE,
        metadata={"help": "How we transverse a graph"}
    )

    shuffle_neighbor_order: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether we shuffle the order of neighbors"}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=True,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )



def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(True)

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = \
        parser.parse_args_into_dataclasses(config.hf_args, config.hf_args_filename)

    gs.initialize(ip_config=config.ip_config, backend=config.backend)

    rt_profiler.init(config.profile_path, rank=gs.get_rank())
    sys_tracker.init(config.verbose, rank=gs.get_rank())
    device = setup_device(config.local_rank)


    train_data = GSgnnNodeTrainData(config.graph_name,
                                    config.part_config,
                                    train_ntypes=config.target_ntype,
                                    node_feat_field=config.node_feat_name,
                                    label_field=config.label_field)

    dataloader = GSlmHatNodeDataLoader(train_data,
                                       prepare_input_fn=prepare_hat_node_centric,
                                       target_idx=train_data.train_idxs,
                                       fanout=config.fanout,
                                       batch_size=config.batch_size,
                                       max_sequence_length=data_args.max_seq_length,
                                       max_sentence_length=data_args.max_sentence_length,
                                       pin_memory=training_args.dataloader_pin_memory,
                                       num_workers=training_args.dataloader_num_workers,
                                       transverse_format=data_args.transverse_format,
                                       shuffle_neighbor_order=data_args.shuffle_neighbor_order)


    val_dataloader = None
    test_dataloader = None
    # we don't need fanout for full-graph inference
    fanout = config.eval_fanout if config.use_mini_batch_infer else []
    if len(train_data.val_idxs) > 0:
        val_dataloader = GSlmHatNodeDataLoader(train_data,
                                       prepare_input_fn=prepare_hat_node_centric,
                                       target_idx=train_data.val_idxs,
                                       fanout=config.eval_fanout,
                                       batch_size=config.batch_size,
                                       max_sequence_length=data_args.max_seq_length,
                                       max_sentence_length=data_args.max_sentence_length,
                                       pin_memory=training_args.dataloader_pin_memory,
                                       num_workers=0,
                                       transverse_format=data_args.transverse_format,
                                       shuffle_neighbor_order=False)
    if len(train_data.test_idxs) > 0:
        test_dataloader = GSlmHatNodeDataLoader(train_data,
                                       prepare_input_fn=prepare_hat_node_centric,
                                       target_idx=train_data.test_idxs,
                                       fanout=config.eval_fanout,
                                       batch_size=config.batch_size,
                                       max_sequence_length=data_args.max_seq_length,
                                       max_sentence_length=data_args.max_sentence_length,
                                       pin_memory=training_args.dataloader_pin_memory,
                                       num_workers=0,
                                       transverse_format=data_args.transverse_format,
                                       shuffle_neighbor_order=False)

    model = gs.create_builtin_hat_model(train_data.g, config, train_task=True)

    trainer = GSgnnHATMasedLMTrainer(model, gs.get_rank(),
        topk_model_to_save=config.topk_model_to_save)

    trainer.fit(train_data, config.num_epochs,
                train_loader=dataloader,
                val_loader=val_dataloader,
                test_loader=test_dataloader)

def generate_parser():
    """ Generate an argument parser
    """
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    arg_parser=generate_parser()

    args = arg_parser.parse_args()
    print(args)
    main(args)
