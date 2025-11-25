from __future__ import annotations
import threading
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING
import traceback, logging

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from src.events import (
    EvalDoneEvent,
    EvalStartEvent,
    EvalErrorEvent,
    GPUEvent,
    PredictStartEvent,
    PredictErrorEvent,
    PredictDoneEvent,
    TrainEvent,
)
from src.models import ModelsFactory
from src.models.Types import ModelBatchPrediction
from src.controllers.DBController import DBController
from src.controllers.DashboardController import DashboardController
from src.controllers.GPUController import GPUController
from src.controllers.ModelSavingController import ModelSavingController
from src.controllers.ModelStateController import ModelStateController
from src.controllers.StreamEventsController import StreamEventsController

if TYPE_CHECKING:
    from src.models.Types import BaseTrainConfig as TrainConfigInput
else:
    from src.models import TrainConfigInput

BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = BASE_DIR / "gui"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

db = DBController()
dash = DashboardController()
saver = ModelSavingController()
stream = StreamEventsController()
model_state = ModelStateController(stream)

ModelsFactory(saver, db, dash)


def _train_worker(model_name: str, config: TrainConfigInput) -> None:
    state = model_state.state_for(model_name)
    try:
        train, val = db.get_datasets()

        model = ModelsFactory.get_model_by_name(
            name=model_name,
            on_load_progress=model_state.get_on_load_progress_callback(model_name),
        )
        state.running = True
        state.progress = 0
        stream.emit(
            TrainEvent(
                model=model_name,
                state=state,
            )
        )

        model.train(
            config=config,
            train_data=train,
            val_data=val,
            on_epoch=model_state.get_on_epoch_callback(model_name),
        )
        state.running = False
        state.progress = 1.0
        stream.emit(
            TrainEvent(
                model=model_name,
                state=state,
            )
        )
    finally:
        state.running = False


def _run_prediction(model_name: str, query: str, threshold: float = 0.5) -> None:
    try:
        model = ModelsFactory.get_model_by_name(
            name=model_name,
            on_load_progress=model_state.get_on_load_progress_callback(model_name),
        )
        out = model.predict_one(query, threshold=threshold)
        if out.error:
            stream.emit(
                PredictErrorEvent(
                    model=model_name,
                    query=query,
                    error=out.hint,
                )
            )
        else:
            stream.emit(
                PredictDoneEvent(
                    model=model_name,
                    prediction=out,
                )
            )
    except Exception as e:
        logging.exception("Predict failed")
        etype = type(e).__name__
        last = traceback.extract_tb(e.__traceback__)[-1] if e.__traceback__ else None
        where = f"{last.filename}:{last.lineno}" if last else "unknown"
        stream.emit(
            PredictErrorEvent(
                model=model_name,
                query=query,
                error=f"{etype}: {e} @ {where}",
            )
        )


router = APIRouter()


@router.get("/tailwind.js")
def tailwind_js():
    return FileResponse(
        TEMPLATES_DIR / "tailwind.cdn.js", media_type="application/javascript"
    )


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    models = ModelsFactory.get_names_list()
    default_model = models[0] if models else ""
    init_load = model_state.state_for(default_model)

    thr_defaults = {}
    for m in models:
        cls = ModelsFactory.get_model_class(m)
        thr_defaults[m] = float(cls.get_default_threshold())
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": models,
            "default_model": default_model,
            "init_load_progress": init_load.load_progress,
            "thr_defaults": thr_defaults,
        },
    )


@router.get("/events")
async def events():
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        stream.event_stream(), media_type="text/event-stream", headers=headers
    )


@router.get("/state")
def state(model_name: str = Query(...)) -> Dict[str, Any]:
    s = model_state.state_for(model_name)
    return {
        "load": {
            "running": s.load_running,
            "progress": s.load_progress,
        },
        "train": {
            "running": s.running,
            "epoch": s.epoch,
            "train_loss": s.train_loss,
            "val_loss": s.val_loss,
            "train_acc": s.train_auc,
            "val_acc": s.val_auc,
            "progress": s.progress,
        },
    }


@router.post("/train/start")
def train_start(cfg: TrainConfigInput):
    model_name: str = cfg.model_name
    state = model_state.state_for(model_name)
    if state.running:
        return {"started": False}
    if state.load_running:
        stream.emit(
            TrainEvent(
                model=model_name,
                state=state,
            )
        )
        raise HTTPException(status_code=409, detail={"error": "model_loading"})
    t = threading.Thread(
        target=_train_worker,
        args=(
            model_name,
            cfg,
        ),
        daemon=True,
    )
    t.start()


@router.post("/predict/start")
def predict_start(
    model_name: str = Query(...),
    q: str = Query(...),
    threshold: Optional[float] = Query(None),
):
    if model_name not in ModelsFactory.get_names_list():
        raise HTTPException(
            status_code=404, detail={"error": "unknown_model", "model_name": model_name}
        )

    s = model_state.state_for(model_name)
    if s.running:
        stream.emit(
            PredictErrorEvent(
                model=model_name,
                query=q,
                error="model_training",
            )
        )
        raise HTTPException(
            status_code=409,
            detail={"error": "model_training", "model_name": model_name},
        )

    stream.emit(
        PredictStartEvent(
            model=model_name,
            query=q,
        )
    )
    t = threading.Thread(
        target=_run_prediction, args=(model_name, q, threshold), daemon=True
    )
    t.start()


@router.get("/predict/val")
def predict_val(model_name: str = Query(...), threshold: float = Query(None)):
    if model_name not in ModelsFactory.get_names_list():
        raise HTTPException(
            status_code=404, detail={"error": "unknown_model", "model_name": model_name}
        )

    s = model_state.state_for(model_name)
    if s.running:
        stream.emit(
            EvalErrorEvent(
                model=model_name,
                error="model_training",
            )
        )
        raise HTTPException(
            status_code=409,
            detail={"error": "model_training", "model_name": model_name},
        )

    try:
        test_data = db.get_test_dataset()
        if not test_data:
            stream.emit(
                EvalErrorEvent(
                    model=model_name,
                    error="empty_validation_set",
                )
            )
        model = ModelsFactory.get_model_by_name(
            name=model_name,
            on_load_progress=model_state.get_on_load_progress_callback(model_name),
        )

        stream.emit(
            EvalStartEvent(
                model=model_name,
                total=len(test_data),
                threshold=threshold,
            )
        )

        on_step = model_state.get_on_eval_callback(model_name)
        report: ModelBatchPrediction = model.predict_batch(
            test_data, evaluation_callback=on_step, threshold=threshold
        )
        stream.emit(
            EvalDoneEvent(
                model=model_name,
                report=report,
                threshold=threshold,
            )
        )

    except Exception as e:
        stream.emit(
            EvalErrorEvent(
                model=model_name,
                error=f"{type(e).__name__}: {e}",
            )
        )
        raise HTTPException(
            status_code=500, detail={"error": "eval_failed", "message": str(e)}
        )


@router.post("/gpu/check")
def gpu_check():
    rep = GPUController().probe()
    stream.emit(
        GPUEvent(
            model=None,
            report=rep,
        )
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Mini ORG/LOC API")
    app.include_router(router)
    return app


app = create_app()
