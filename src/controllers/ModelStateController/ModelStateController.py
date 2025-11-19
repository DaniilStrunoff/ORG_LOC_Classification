from dataclasses import dataclass
from functools import partial
from typing import Protocol

from src.controllers.StreamEventsController import StreamEventsController
from src.controllers.IController import IController
from src.models.Types import ModelPrediction
from src.events import LoadEvent, TrainEvent, EvalStepEvent


@dataclass(slots=True)
class ModelState:
    running: bool = False
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_auc: float = 0.0
    val_auc: float = 0.0
    progress: float = 0.0
    load_running: bool = False
    load_progress: float = 0.0

    def update_state(self, **kwargs: float | None) -> None:
        for k, v in kwargs.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, float(v))


class OnEpoch(Protocol):
    def __call__(
        self,
        e: int,
        train_loss: float,
        val_loss: float,
        train_auc: float,
        val_auc: float,
        progress: float,
    ) -> None: ...


class OnLoadProgress(Protocol):
    def __call__(self, current: int, total: int) -> None: ...


class OnEvalStep(Protocol):
    def __call__(self, i: int, total: int, item: ModelPrediction) -> None: ...


class ModelStateController(IController):
    _stream: StreamEventsController
    _state_by_model: dict[str, ModelState]

    def __init__(self, stream: StreamEventsController):
        self._stream = stream
        self._state_by_model = {}

    def state_for(self, model_name: str) -> ModelState:
        st = self._state_by_model.get(model_name)
        if st is None:
            st = ModelState()
            self._state_by_model[model_name] = st
        return st

    def _on_load_progress(self, current: int, total: int, *, model_name: str) -> None:
        s = self.state_for(model_name)
        p = float(current / total) if total > 0 else 0.0
        s.load_progress = p
        s.load_running = p < 1.0
        self._stream.emit(LoadEvent(model=model_name, progress=p))

    def _on_epoch(
        self,
        e: int,
        train_loss: float,
        val_loss: float,
        train_auc: float,
        val_auc: float,
        progress: float,
        *,
        model_name: str,
    ) -> None:
        s = self.state_for(model_name)
        s.epoch = e
        s.update_state(
            train_loss=train_loss,
            val_loss=val_loss,
            train_auc=train_auc,
            val_auc=val_auc,
            progress=progress,
        )
        s.running = True
        self._stream.emit(TrainEvent(model=model_name, state=s))

    def _on_eval_step(
        self,
        i: int,
        total: int,
        prediction: ModelPrediction,
        *,
        model_name: str,
    ) -> None:
        self._stream.emit(
            EvalStepEvent(
                model=model_name,
                i=i,
                total=total,
                prediction=prediction,
            )
        )

    def get_on_epoch_callback(self, model_name: str) -> OnEpoch:
        return partial(self._on_epoch, model_name=model_name)

    def get_on_load_progress_callback(self, model_name: str) -> OnLoadProgress:
        return partial(self._on_load_progress, model_name=model_name)

    def get_on_eval_callback(self, model_name: str) -> OnEvalStep:
        return partial(self._on_eval_step, model_name=model_name)
