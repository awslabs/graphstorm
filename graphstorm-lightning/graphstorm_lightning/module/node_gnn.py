from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import graphstorm_lightning as gsl
from graphstorm import create_builtin_node_gnn_model
from graphstorm.model import GSgnnNodeModelInterface
from graphstorm.model.gnn import GSOptimizer
from graphstorm.run.gsgnn_np.gsgnn_np import get_evaluator
from graphstorm.trainer.np_trainer import do_full_graph_inference
from typing_extensions import override


class GSgnnNodeModel(pl.LightningModule):
    def __init__(
        self,
        datamodule: pl.LightningDataModule,
        config: Dict[str, Any],
        max_grad_norm: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["data"])
        self.datamodule = datamodule
        self.model: GSgnnNodeModelInterface
        self.optimizer: GSOptimizer
        self.val_preds = {}
        self.val_labels = {}

    @override
    def configure_model(self) -> None:
        if hasattr(self, "model"):
            return
        config = gsl.utils.get_config(self.trainer, self.hparams.config)
        self.val_fanout = config.eval_fanout
        train_data = self.datamodule.gnn
        self.model = create_builtin_node_gnn_model(train_data.g, config, train_task=True)
        self.evaluator = get_evaluator(config)

    @override
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        # from source
        input_nodes, lbl, blocks, input_feats = batch
        # torch.distributed might move this tensor to GPU
        input_nodes = pl.utilities.move_data_to_device(input_nodes, "cpu")
        loss = self.model(blocks, input_feats, None, lbl, input_nodes)
        self.manual_backward(loss)

        # clip gradient
        if self.hparams.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_grad_norm)

        # accumulate gradients of N batches
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.log("train_loss", loss, prog_bar=True)
        return loss

    @override
    def on_validation_epoch_start(self) -> None:
        model = self.model
        data = self.datamodule.gnn

        # from source
        self.val_emb = do_full_graph_inference(model, data, fanout=self.val_fanout)

    @override
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        model = self.model
        device = self.device
        labels = self.val_labels
        preds = self.val_preds
        emb = self.val_emb

        # from source
        input_nodes, lbl, _, _ = batch
        # torch.distributed might move this tensor to GPU
        input_nodes = pl.utilities.move_data_to_device(input_nodes, "cpu")
        for ntype, in_nodes in input_nodes.items():
            if isinstance(model.decoder, torch.nn.ModuleDict):
                assert ntype in model.decoder, f"Node type {ntype} not in decoder"
                decoder = model.decoder[ntype]
            else:
                decoder = model.decoder
            pred = decoder.predict(emb[ntype][in_nodes].to(device))
            if ntype in preds:
                preds[ntype].append(pred.cpu())
            else:
                preds[ntype] = [pred.cpu()]
            if ntype in labels:
                labels[ntype].append(lbl[ntype])
            else:
                labels[ntype] = [lbl[ntype]]

    @override
    def on_validation_epoch_end(self) -> None:
        self.val_emb = None
        total_steps = self.trainer.global_step
        preds = self.val_preds
        labels = self.val_labels

        # from source
        val_pred = {}
        val_label = {}
        for ntype, ntype_pred in preds.items():
            val_pred[ntype] = torch.cat(ntype_pred)
        for ntype, ntype_label in labels.items():
            val_label[ntype] = torch.cat(ntype_label)
        labels.clear()
        preds.clear()

        # TODO(wlcong) we only support node prediction on one node type for evaluation now
        assert len(val_label) == 1, "We only support prediction on one node type for now."
        ntype = list(val_label.keys())[0]
        # We need to have val and label (test and test label) data in GPU
        # when backend is nccl, as we need to use nccl.all_reduce to exchange
        # data between GPUs
        val_pred = val_pred[ntype]
        val_label = val_label[ntype]
        val_score, test_score = self.evaluator.evaluate(val_pred, None, val_label, None, total_steps)
        for metric in self.evaluator.metric:
            self.log(f"val_{metric}", val_score[metric])
            best_val_score = self.evaluator.best_val_score
            self.log(f"best_val_{metric}", best_val_score[metric])

    @override
    def configure_optimizers(self) -> pl.utilities.types.OptimizerLRScheduler:
        self.optimizer = self.model.create_optimizer()
        return []
