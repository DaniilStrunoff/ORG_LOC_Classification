import importlib, pkgutil
from typing import Any, cast

from src.models.ModelsFactory import ModelsFactory
from src.models.IModel import IModel


__all__ = ["ModelsFactory", "IModel", "TrainConfigInput"]


def _auto_import_all_submodules():
    for _, modname, ispkg in pkgutil.walk_packages(__path__, __name__ + "."):
        if ispkg:
            continue
        importlib.import_module(modname)


_auto_import_all_submodules()

TrainConfigInput = cast(Any, ModelsFactory.build_train_config_union())
