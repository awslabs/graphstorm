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

    GraphStorm trainer for pre-training an HAT model on graph data.
"""
import time
import resource
import dataclasses
import torch as th

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

from ..model.gnn import GSgnnModelBase

from .gsgnn_trainer import GSgnnTrainer

from ..utils import sys_tracker
from ..utils import rt_profiler

class GSgnnHATMasedLMTrainer(GSgnnTrainer):
    """ HAT mask language mask pre-training Trainer

    Parameters
    ----------
    model : GSgnnLinkPredictionModelBase
        The HAT model for pre-training
    rank : int
        The rank.
    topk_model_to_save : int
        The top K model to save.
    """
    def __init__(self, model, rank, topk_model_to_save):
        super(GSgnnHATMasedLMTrainer, self).__init__(model, rank, topk_model_to_save)
        assert isinstance(model, GSgnnModelBase), \
                "The input model is not a GSgnnModel"

    def fit(self, train_dataset, num_epochs,
            eval_dataset=None,            # pylint: disable=unused-argument
            test_loader=None,           # pylint: disable=unused-argument
            use_mini_batch_infer=True,      # pylint: disable=unused-argument
            save_model_path=None,
            save_model_frequency=None,
            save_perf_results_path=None,
            **kwargs):
        """ The fit function for link prediction.

        Parameters
        ----------
        train_loader : GSgnnLinkPredictionDataLoader
            The mini-batch sampler for training.
        num_epochs : int
            The max number of epochs to train the model.
        val_loader : GSgnnLinkPredictionDataLoader
            The mini-batch sampler for computing validation scores. The validation scores
            are used for selecting models.
        test_loader : GSgnnLinkPredictionDataLoader
            The mini-batch sampler for computing test scores.
        use_mini_batch_infer : bool
            Whether or not to use mini-batch inference.
        save_model_path : str
            The path where the model is saved.
        save_model_frequency : int
            The number of iteration to train the model before saving the model.
        save_perf_results_path : str
            The path of the file where the performance results are saved.
        edge_mask_for_gnn_embeddings : str
            The mask that indicates the edges used for computing GNN embeddings for model
            evaluation. By default, we use the edges in the training graph to compute
            GNN embeddings for evaluation.
        freeze_input_layer_epochs: int
            Freeze input layer model for N epochs. This is commonly used when
            the input layer contains language models.
            Default: 0, no freeze.
        """

        # collect args from transfomers.TrainingArguments
        keys = {f.name for f in dataclasses.fields(TrainingArguments) if f.init}
        train_kwargs = {key: val \
            for key, val in vars(kwargs).items() if key in keys}

        training_args = TrainingArguments(
            train_kwargs
        )

        training_args.set_dataloader()

        # https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer.py#L491-L495

        # https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer.py#L281C64-L282C95

        # Initialize our transformers.Trainer
        trainer = GsHuggingfaceTrainer(
            gs_dataloader=gs_dataloader,
            device=device,
            model=self._model,
            args=training_args,
            train_dataset=train_dataset
            eval_dataset=None, # GraphStorm store eval and test set in train_dataset
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
