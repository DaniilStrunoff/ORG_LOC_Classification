from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional, Tuple, cast
import time
import numpy as np
from pydantic import Field
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, TaskType, LoraConfig

from src.models.Types import BaseTrainConfig, ModelPrediction, Label
from src.models import IModel, ModelsFactory
from src.controllers.HFModelController import HFModelController
from src.controllers.DBController import DBController
from src.controllers.DashboardController import DashboardController
from src.controllers.ModelSavingController import ModelSavingController
from src.controllers.ModelStateController import OnLoadProgress, OnEpoch


MODEL_NAME: str = "bert_lora"


class BERTLoRATrainConfig(BaseTrainConfig):
    model_name: Literal["bert_lora"] = MODEL_NAME  # type: ignore
    r: int = Field(16, title="Matrix rank", ge=2, le=32)
    lora_alpha: int = Field(16, title="LoRa alpha", ge=1, le=1000)
    lora_dropout: float = Field(0.1, title="LoRA dropout", ge=0, le=1)
    epochs: int = Field(250, title="Epochs", ge=1, le=10000)
    batch_size: int = Field(16, title="Batch size", ge=1, le=1000)
    lr: float = Field(1e-4, title="Learning rate", ge=0, le=0.1)
    weight_decay: float = Field(0.01, title="Weight decay", ge=0, le=0.5)
    max_len: int = Field(256, title="Max sequence length", ge=1, le=10000)
    target_modules: List[Literal["query_proj", "value_proj"]] = Field(
        default=["query_proj", "value_proj"],
        title="Target modules",
        description="Which submodules to apply LoRA to",
    )


Sample = Tuple[str, int]


@dataclass
class TextLabelDataset(Dataset[Sample]):
    texts: List[str]
    labels: np.ndarray[tuple[Any, ...], np.dtype[np.int64]]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        return self.texts[i], int(self.labels[i])


def _make_collate(
    tokenizer: PreTrainedTokenizerBase,
    max_len: int = 256,
) -> Callable[[List[Sample]], BatchEncoding]:
    def collate(batch: List[Sample]) -> BatchEncoding:
        texts = [t for t, _ in batch]
        y = torch.tensor([y for _, y in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc["labels"] = y
        return enc

    return collate


class BERT_LoRA(IModel[BERTLoRATrainConfig]):
    BASE_MODEL = "microsoft/mdeberta-v3-base"
    TINY_MODEL = "hf-internal-testing/tiny-random-deberta-v2"
    NUM_LABELS = 2
    ADAPTER_FILE = "lora.pt"

    def __init__(
        self,
        saver: Optional[ModelSavingController] = None,
        db: Optional[DBController] = None,
        dash: Optional[DashboardController] = None,
        model_name: str = MODEL_NAME,
        on_load_progress: Optional[OnLoadProgress] = None,
    ) -> None:
        self.saver = saver or ModelSavingController()
        self.db = db or DBController()
        self.dash = dash or DashboardController()
        self.model_name = model_name
        self.labels = {0: "ORG", 1: "LOC"}
        self.label2id = {v: k for k, v in self.labels.items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_model_loader = HFModelController()
        self.tokenizer = base_model_loader.load_model(
            repo_id=self.BASE_MODEL,
            tiny_repo_id=self.TINY_MODEL,
            model_class=AutoTokenizer,
            progress=on_load_progress,
            use_fast=False,
        )
        self.base = base_model_loader.load_model(
            repo_id=self.BASE_MODEL,
            tiny_repo_id=self.TINY_MODEL,
            model_class=AutoModelForSequenceClassification,
            progress=on_load_progress,
            num_labels=self.NUM_LABELS,
            id2label=self.labels,
            label2id=self.label2id,
            return_dict=True,
        )
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.00)
        self.ready = False

    def _build_loader(
        self,
        pairs: List[tuple[str, Label]],
        max_len: int,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
    ) -> DataLoader[Sample]:
        xs = [t for t, _ in pairs]
        ys = np.array([self.label2id[l.upper()] for _, l in pairs], dtype=np.int64)
        ds = TextLabelDataset(xs, ys)
        if not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise ValueError("tokenizer is not PreTrainedTokenizerBase!")
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_make_collate(self.tokenizer, max_len),
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
        )

    @torch.no_grad()
    def _eval_loader(self, dloader: DataLoader[Sample]) -> Tuple[float, float]:
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_items = 0
        y_true: List[torch.Tensor] = []
        y_scores: List[torch.Tensor] = []
        pos = self.label2id["ORG"]
        for batch in dloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model(
                **{k: v for k, v in batch.items() if k != "labels"}
            ).logits.float()
            loss = self.criterion(logits, batch["labels"])
            probs = torch.softmax(logits, dim=-1)
            bs = batch["labels"].size(0)
            total_loss += float(loss.item()) * bs
            total_items += bs
            y_true.append(batch["labels"].detach().cpu())
            y_scores.append(probs[:, pos].detach().cpu())
        if was_training:
            self.model.train()
        avg_loss = (total_loss / total_items) if total_items else 0.0
        if total_items:
            yt = torch.cat(y_true).numpy()
            ys = torch.cat(y_scores).numpy()
            auc = float(roc_auc_score((yt == pos).astype(np.int32), ys))
        else:
            auc = float("nan")
        return float(avg_loss), float(auc)

    def train(
        self,
        config: BERTLoRATrainConfig,
        train_data: List[tuple[str, Label]],
        val_data: List[tuple[str, Label]],
        on_epoch: OnEpoch = lambda *args, **kwargs: None,
    ) -> None:
        if not train_data or not val_data:
            raise RuntimeError("Empty train/val data")

        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=cast(List[str], config.target_modules),
            modules_to_save=["classifier"],
        )
        if not isinstance(self.base, PreTrainedModel):
            raise ValueError(f"Base model should be PreTrainedModel")
        self.model = get_peft_model(self.base, lora_cfg).to(self.device)

        dl_tr = self._build_loader(
            train_data, config.max_len, config.batch_size, shuffle=True, drop_last=True
        )
        dl_va = self._build_loader(
            val_data, config.max_len, max(64, config.batch_size), shuffle=False
        )

        self.model.train()
        for name, p in self.model.named_parameters():
            if "lora_" in name or "modules_to_save" in name:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

        optim = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6,
        )
        sched = get_linear_schedule_with_warmup(
            optim,
            max(1, int(0.1 * config.epochs * len(dl_tr))),
            config.epochs * len(dl_tr),
        )

        run_id = self.db.create_training_run()
        try:
            self.dash.create_training_dashboard(run_id)
        except Exception:
            pass

        best_auc = -1.0
        for e in range(1, config.epochs + 1):
            epoch_loss_sum, epoch_items = 0.0, 0
            for batch in dl_tr:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(
                    **{k: v for k, v in batch.items() if k != "labels"}
                ).logits.float()
                loss = self.criterion(logits, batch["labels"])
                loss.backward()
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                bs = batch["labels"].size(0)
                epoch_loss_sum += float(loss.item()) * bs
                epoch_items += bs

            train_loss, train_auc = self._eval_loader(dl_tr)
            val_loss, val_auc = self._eval_loader(dl_va)

            try:
                self.db.log_training_metric(
                    run_id,
                    e,
                    float(train_loss),
                    float(val_loss),
                    float(train_auc),
                    float(val_auc),
                )
            except Exception:
                pass

            on_epoch(
                e,
                float(train_loss),
                float(val_loss),
                float(train_auc),
                float(val_auc),
                e / config.epochs,
            )

            if val_auc > best_auc + 1e-4:
                print(self.ADAPTER_FILE, val_auc)
                best_auc = val_auc
                peft_sd = {}
                for k, v in self.model.state_dict().items():
                    if "lora_" in k or "modules_to_save" in k:
                        peft_sd[k] = v.detach().cpu()
                self.saver.save_torch_state_dict(
                    self.model_name, peft_sd, filename=self.ADAPTER_FILE
                )

        best_state = self.saver.load_torch_state_dict(
            self.model_name, filename=self.ADAPTER_FILE, map_location=str(self.device)
        )
        self.model.load_state_dict(best_state, strict=False)
        self.model.eval()
        self.ready = True

    def predict_one(
        self, text: str, threshold: Optional[float] = None
    ) -> ModelPrediction:
        if not self.ready:
            return ModelPrediction(
                text=text,
                gold="UNKNOWN",
                label="UNKNOWN",
                confidence=0.0,
                latency_ms=0.0,
                error=True,
                hint="call train() first",
            )
        t0 = time.time()
        x = self.tokenizer(
            [text], return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(**x).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        p_org = float(probs[self.label2id["ORG"]])
        thr = (
            float(threshold) if threshold is not None else self.get_default_threshold()
        )
        label = "ORG" if p_org >= thr else "LOC"
        confidence = p_org if label == "ORG" else 1.0 - p_org
        return ModelPrediction(
            text=text,
            gold="UNKNOWN",
            label=label,
            confidence=confidence,
            latency_ms=round((time.time() - t0) * 1000, 2),
            error=False,
            hint="",
        )


ModelsFactory.register_model(MODEL_NAME, BERT_LoRA)
