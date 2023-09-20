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

from datasets import load_metric

from ..model.gnn import GSgnnModelBase
from .gsgnn_trainer import GSgnnTrainer

from ..utils import sys_tracker
from ..utils import rt_profiler

from ..model.graph_transformer import GsHuggingfaceTrainer
from ..model.graph_transformer import preprocess_logits_for_mlm_metrics

class GSgnnHATMasedLMTrainer(GSgnnTrainer):
    """ HAT mask language mask pre-training Trainer

    Parameters
    ----------
    model : GSgnnLinkPredictionModelBase
        The HAT model for pre-training
    """
    def __init__(self, model):
        self._model = model

    def fit(self, train_dataset,
            training_args,
            train_loader,
            val_loader=None,
            test_loader=None):
        """ The fit function for link prediction.

        Parameters
        ----------
        train_loader : GSgnnLinkPredictionDataLoader
            The mini-batch sampler for training.
        training_args: dict
            Args for Huggingface trainer
        train_loader : GSlmHatNodeDataLoader
            The mini-batch sampler for training.
        val_loader : GSlmHatNodeDataLoader
            The mini-batch sampler for computing validation scores. The validation scores
            are used for selecting models.
        test_loader : GSlmHatNodeDataLoader
            The mini-batch sampler for computing test scores.
        """
        metric = load_metric("accuracy")

        def compute_mlm_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

        # Initialize our transformers.Trainer
        trainer = GsHuggingfaceTrainer(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None, # GraphStorm store eval and test set in train_dataset
            compute_metrics=compute_mlm_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_mlm_metrics)

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint

            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model(training_args.output_dir)  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

