from src.controllers.IController import IController
from typing import Any, Optional
from pathlib import Path
import torch
import os


class ModelSavingController(IController):
    def __init__(self, root: Optional[str | Path] = None) -> None:
        root = root or os.getenv("MODEL_STORE_ROOT", "./model_store")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, model_name: str) -> Path:
        d = self.root / model_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_torch_state_dict(
        self, model_name: str, state_dict: Any, filename: str = "model.pt"
    ) -> Path:
        path = self._model_dir(model_name) / filename
        torch.save(state_dict, path)
        print(f"Model saved: {path}")
        return path

    def load_torch_state_dict(
        self, model_name: str, filename: str = "model.pt", map_location: str = "cpu"
    ) -> Any:
        path = self._model_dir(model_name) / filename
        print(f"Model loaded: {path}")
        return torch.load(str(path), map_location=map_location)
