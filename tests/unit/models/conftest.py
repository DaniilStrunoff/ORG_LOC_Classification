from __future__ import annotations
import pytest
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)
from typing import Any, TYPE_CHECKING
from src.controllers.HFModelController import (
    HFModelController,
    ModelFactory,
    TokenizerFactory,
)

if TYPE_CHECKING:
    from src.controllers.ModelStateController import OnLoadProgress

original_load_model = HFModelController.load_model


@pytest.fixture(autouse=True)
def patch_hf_models_for_unit_models(monkeypatch: pytest.MonkeyPatch):
    def fake_load_model(
        self: HFModelController,
        repo_id: str,
        tiny_repo_id: str,
        model_class: type[ModelFactory] | type[TokenizerFactory] = AutoModel,
        revision: str = "main",
        token: str | None = None,
        with_tokenizer: bool = False,
        progress: OnLoadProgress | None = None,
        **from_pretrained_kwargs: Any,
    ) -> PreTrainedModel | PreTrainedTokenizerBase:
        big_cfg = AutoConfig.from_pretrained(repo_id, force_download=True)  # type: ignore[reportUnknownMemberType]
        tiny_cfg = AutoConfig.from_pretrained(tiny_repo_id, force_download=True)  # type: ignore[reportUnknownMemberType]
        if not isinstance(big_cfg, PretrainedConfig):
            raise ValueError(f"Original repo id congig is none!")
        if not isinstance(tiny_cfg, PretrainedConfig):
            raise ValueError(f"Tiny repo id congig is none!")
        if big_cfg.model_type != tiny_cfg.model_type:
            raise ValueError(
                f"tiny_repo_id={tiny_repo_id} has model_type='{tiny_cfg.model_type}', "
                f"but repo_id={repo_id} has model_type='{big_cfg.model_type}'. "
                "They MUST match."
            )
        return original_load_model(
            self,
            tiny_repo_id,
            tiny_repo_id,
            model_class,
            revision,
            token,
            with_tokenizer,
            progress,
            **from_pretrained_kwargs,
        )

    monkeypatch.setattr(HFModelController, "load_model", fake_load_model, raising=True)
