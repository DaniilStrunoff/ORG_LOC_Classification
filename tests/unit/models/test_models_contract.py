from typing import Any, Dict, List

import pytest

from src.models import ModelsFactory
from src.models.Types import ModelPrediction, Label


def get_model_names() -> list[str]:
    return ModelsFactory.get_names_list()


@pytest.mark.parametrize("model_name", get_model_names())
def test_model_can_train_and_predict(model_name: str):
    model_cls = ModelsFactory.get_model_class(model_name)
    model = model_cls()

    train_data: list[tuple[str, Label]] = [
        ("сбербанк", "ORG"),
        ("москва", "LOC"),
        ("тинькофф банк", "ORG"),
        ("санкт петербург", "LOC"),
        ("втб", "ORG"),
        ("новосибирск", "LOC"),
        ("газпром", "ORG"),
        ("екатеринбург", "LOC"),
        ("ростелеком", "ORG"),
        ("нижний новгород", "LOC"),
        ("лукойл", "ORG"),
        ("красноярск", "LOC"),
        ("яндекс", "ORG"),
        ("казань", "LOC"),
        ("магнит", "ORG"),
        ("сочи", "LOC"),
        ("роснефть", "ORG"),
        ("самара", "LOC"),
        ("мтс", "ORG"),
        ("уфа", "LOC"),
    ]

    val_data: list[tuple[str, Label]] = [
        ("озон", "ORG"),
        ("пермь", "LOC"),
        ("мэйл ру групп", "ORG"),
        ("омск", "LOC"),
        ("вайлдберриз", "ORG"),
        ("тюмень", "LOC"),
        ("альфа банк", "ORG"),
        ("волгоград", "LOC"),
    ]

    epochs_auc: List[float] = []

    def on_epoch(
        e: int,
        train_loss: float,
        val_loss: float,
        train_auc: float,
        val_auc: float,
        progress: float,
    ) -> None:
        epochs_auc.append(val_auc)

    train_kwargs: Dict[str, Any] = {}
    config = ModelsFactory.get_config_class(model_name)
    if "epochs" in config.model_fields:
        train_kwargs["epochs"] = 2
    if "batch_size" in config.model_fields:
        train_kwargs["batch_size"] = 2
    if "max_len" in config.model_fields:
        train_kwargs["max_len"] = 32

    try:
        model.train(
            config=config(**train_kwargs),
            train_data=train_data,
            val_data=val_data,
            on_epoch=on_epoch,
        )
    except NotImplementedError:
        pytest.skip(f"train not implemented for model {model_name}")

    assert epochs_auc

    single = model.predict_one("Sberbank", threshold=0.5)
    assert isinstance(single, ModelPrediction)
    assert not single.error

    seen_steps: list[int] = []

    def on_step(i: int, total: int, item: ModelPrediction):
        seen_steps.append(i)
        assert isinstance(item, ModelPrediction)

    report = model.predict_batch(
        data=val_data,
        evaluation_callback=on_step,
        threshold=0.5,
    )

    assert report.total == len(val_data)
    assert len(seen_steps) == len(val_data)
    if len(epochs_auc) > 1:
        assert epochs_auc[
            -1
        ] == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
            report.auc_roc, rel=1e-6
        )
    if len(epochs_auc) == 1:
        assert epochs_auc[
            -1
        ] == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
            1, rel=1e-6
        )
