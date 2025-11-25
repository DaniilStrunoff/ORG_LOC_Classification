from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Protocol
from sklearn.metrics import roc_auc_score  # type: ignore
import torch

from src.models.Types import (
    ModelPrediction,
    ModelBatchPrediction,
    Label,
    TBaseTrainConfig,
)
from src.controllers.DBController import DBController
from src.controllers.DashboardController import DashboardController
from src.controllers.ModelSavingController import ModelSavingController
from src.controllers.ModelStateController import OnLoadProgress, OnEpoch, OnEvalStep


class IModel(Generic[TBaseTrainConfig], ABC):
    DECISION_THRESHOLD: float = 0.50

    def __init__(
        self,
        saver: Optional[ModelSavingController] = None,
        db: Optional[DBController] = None,
        dash: Optional[DashboardController] = None,
        model_name: str = "model_name",
        on_load_progress: OnLoadProgress | None = None,
    ) -> None:
        pass

    @abstractmethod
    def train(
        self,
        config: TBaseTrainConfig,
        train_data: List[tuple[str, Label]],
        val_data: List[tuple[str, Label]],
        on_epoch: OnEpoch,
    ) -> None:
        pass

    @abstractmethod
    def predict_one(self, text: str, threshold: float = 0.5) -> ModelPrediction:
        pass

    def predict_batch(
        self,
        data: List[tuple[str, Label]],
        evaluation_callback: OnEvalStep,
        threshold: float = 0.5,
    ) -> ModelBatchPrediction:
        total = len(data)
        covered = 0
        correct = 0
        errors = 0
        y_true_all: List[int] = []
        y_score_all: List[float] = []
        hint_parts: List[str] = []
        for i, (text, gold) in enumerate(data):
            try:
                pred = self.predict_one(text, threshold=threshold)
                pred.gold = gold
                if pred.error:
                    errors += 1
                else:
                    covered += 1
                    if pred.label in ("ORG", "LOC") and pred.label == gold:
                        correct += 1
                    p_org = (
                        pred.confidence
                        if pred.label == "ORG"
                        else 1.0 - pred.confidence
                    )
                    y_true_all.append(1 if gold == "ORG" else 0)
                    y_score_all.append(float(p_org))
            except Exception as e:
                errors += 1
                pred = ModelPrediction(
                    text=text,
                    gold=gold,
                    label="UNKNOWN",
                    confidence=0.0,
                    latency_ms=0.0,
                    error=True,
                    hint=f"inference_failed: {type(e).__name__}: {e}",
                )
            if evaluation_callback:
                evaluation_callback(i, total, pred)
        misclassified = covered - correct
        acc_covered = (correct / covered) if covered else 0.0
        if len(set(y_true_all)) > 1 and len(y_score_all) == len(y_true_all):
            try:
                auc = float(roc_auc_score(y_true_all, y_score_all))
            except Exception:
                auc = 0.0
                hint_parts.append("auc_failed")
        else:
            auc = 0.0
            hint_parts.append("auc_roc_unavailable_single_class")
        hint = ";".join(hint_parts) if hint_parts else ""
        return ModelBatchPrediction(
            total=total,
            covered=covered,
            misclassified=misclassified,
            acc_covered=acc_covered,
            auc_roc=auc,
            error=bool(errors > 0),
            hint=hint,
        )

    @classmethod
    def get_default_threshold(cls) -> float:
        try:
            return float(getattr(cls, "DECISION_THRESHOLD"))
        except Exception:
            return 0.5
