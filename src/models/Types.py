from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeVar

from pydantic import BaseModel


Label = Literal["ORG", "LOC", "UNKNOWN"]


class BaseTrainConfig(BaseModel):
    model_name: str


TBaseTrainConfig = TypeVar("TBaseTrainConfig", bound=BaseTrainConfig)


@dataclass
class ModelPrediction:
    text: str
    gold: Label
    label: Label
    confidence: float
    latency_ms: float
    error: bool = False
    hint: str = ""


@dataclass
class ModelBatchPrediction:
    total: int
    covered: int
    misclassified: int
    acc_covered: float
    auc_roc: float
    error: bool = False
    hint: str = ""
