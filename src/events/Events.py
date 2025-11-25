from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Literal, Optional, Dict, Any, TYPE_CHECKING
from src.controllers.GPUController import GPUReport

if TYPE_CHECKING:
    from src.models.Types import ModelPrediction, ModelBatchPrediction
    from src.controllers.ModelStateController import ModelState


@dataclass(frozen=True, slots=True)
class Event:
    @property
    def type(self) -> str:
        raise NotImplementedError

    @property
    def phase(self) -> Optional[str]:
        return None

    def dump(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type
        ph = self.phase
        if ph is not None:
            d["phase"] = ph
        return d


@dataclass(frozen=True, slots=True)
class LoadEvent(Event):
    model: str = ""
    progress: float = 0.0

    @property
    def type(self) -> Literal["load"]:
        return "load"


@dataclass(frozen=True, slots=True)
class TrainEvent(Event):
    model: str
    running: bool
    progress: float

    def __init__(self, model: str, state: ModelState):
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "running", state.running)
        object.__setattr__(self, "progress", float(state.progress))

    @property
    def type(self) -> Literal["train"]:
        return "train"


@dataclass(frozen=True, slots=True)
class PredictEvent(Event):
    model: str = ""
    query: str = ""
    label: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    @property
    def type(self) -> Literal["predict"]:
        return "predict"

    @property
    def phase(self) -> Literal["start", "done", "error"]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PredictStartEvent(PredictEvent):
    @property
    def phase(self) -> Literal["start"]:
        return "start"


@dataclass(frozen=True, slots=True)
class PredictDoneEvent(PredictEvent):
    def __init__(self, model: str, prediction: ModelPrediction):
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "query", prediction.text)
        object.__setattr__(self, "label", prediction.label)
        object.__setattr__(self, "confidence", float(prediction.confidence))
        object.__setattr__(self, "latency_ms", float(prediction.latency_ms))
        object.__setattr__(
            self,
            "error",
            None if not prediction.error else (prediction.hint or "error"),
        )

    @property
    def phase(self) -> Literal["done"]:
        return "done"


@dataclass(frozen=True, slots=True)
class PredictErrorEvent(PredictEvent):
    def __init__(self, model: str, query: str, error: str):
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "label", None)
        object.__setattr__(self, "confidence", None)
        object.__setattr__(self, "latency_ms", None)

    @property
    def phase(self) -> Literal["error"]:
        return "error"


@dataclass(frozen=True, slots=True)
class EvalEvent(Event):
    model: str = ""

    @property
    def type(self) -> Literal["eval"]:
        return "eval"

    @property
    def phase(self) -> Literal["start", "step", "done", "error"]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class EvalStartEvent(EvalEvent):
    total: int = 0
    threshold: Optional[float] = None

    @property
    def phase(self) -> Literal["start"]:
        return "start"


@dataclass(frozen=True, slots=True)
class EvalStepEvent(EvalEvent):
    i: int = 0
    total: int = 0
    text: str = ""
    gold: str = ""
    pred: str = ""
    org: float = 0.0
    loc: float = 0.0
    latency_ms: float = 0.0
    status: str = ""
    label: str = ""
    confidence: float = 0.0
    error: bool = False

    def __init__(self, model: str, i: int, total: int, prediction: ModelPrediction):
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "i", i)
        object.__setattr__(self, "total", total)
        label = prediction.label
        conf = float(prediction.confidence)
        is_err = bool(prediction.error)
        gold = prediction.gold
        text = prediction.text
        lat = float(prediction.latency_ms)
        status = "ERROR" if is_err else ("CORRECT" if label == gold else "WRONG")
        org = conf if label == "ORG" else 1.0 - conf
        loc = conf if label == "LOC" else 1.0 - conf
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "gold", gold)
        object.__setattr__(self, "pred", label)
        object.__setattr__(self, "org", round(org, 4))
        object.__setattr__(self, "loc", round(loc, 4))
        object.__setattr__(self, "latency_ms", lat)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "confidence", round(conf, 4))
        object.__setattr__(self, "error", is_err)

    @property
    def phase(self) -> Literal["step"]:
        return "step"


@dataclass(frozen=True, slots=True)
class EvalDoneEvent(EvalEvent):
    total: int = 0
    covered: int = 0
    misclassified: int = 0
    acc_covered: float = 0.0
    auc_roc: Optional[float] = None
    threshold: Optional[float] = None

    def __init__(
        self,
        model: str,
        report: ModelBatchPrediction,
        threshold: Optional[float] = None,
    ):
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "total", int(getattr(report, "total")))
        object.__setattr__(self, "covered", int(getattr(report, "covered")))
        object.__setattr__(self, "misclassified", int(getattr(report, "misclassified")))
        object.__setattr__(self, "acc_covered", float(getattr(report, "acc_covered")))
        auc = getattr(report, "auc_roc", None)
        object.__setattr__(self, "auc_roc", None if auc is None else float(auc))
        object.__setattr__(self, "threshold", threshold)

    @property
    def phase(self) -> Literal["done"]:
        return "done"


@dataclass(frozen=True, slots=True)
class EvalErrorEvent(EvalEvent):
    error: str = ""

    @property
    def phase(self) -> Literal["error"]:
        return "error"


@dataclass(frozen=True, slots=True)
class GPUEvent(Event):
    @property
    def type(self) -> Literal["gpu"]:
        return "gpu"

    model: Optional[str] = None
    report: GPUReport = field(default_factory=lambda: GPUReport(gpu_visible=False))
