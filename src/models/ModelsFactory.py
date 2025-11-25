from typing import Dict, List, Optional, Type, get_args, get_origin, Any, Union
from typing_extensions import Annotated
from pydantic import Field
from src.controllers.DBController import DBController
from src.controllers.DashboardController import DashboardController
from src.controllers.ModelSavingController import ModelSavingController
from src.controllers.ModelStateController import OnLoadProgress
from src.models.IModel import IModel
from src.models.Types import BaseTrainConfig


class ModelsFactory:
    _models_dict: Dict[str, IModel[Any]] = {}
    _types_dict: Dict[str, Type[IModel[Any]]] = {}
    _train_cfg_dict: Dict[str, Type[BaseTrainConfig]] = {}
    saver: Optional[ModelSavingController] = None
    db: Optional[DBController] = None
    dash: Optional[DashboardController] = None

    def __init__(
        self,
        saver: Optional[ModelSavingController] = None,
        db: Optional[DBController] = None,
        dash: Optional[DashboardController] = None,
    ) -> None:
        ModelsFactory._models_dict = {}
        ModelsFactory.saver = saver
        ModelsFactory.db = db
        ModelsFactory.dash = dash

    @staticmethod
    def _extract_config_type(
        model_type: Type[IModel[Any]],
    ) -> Optional[Type[BaseTrainConfig]]:
        orig_bases = getattr(model_type, "__orig_bases__", [])
        for base in orig_bases:
            origin = get_origin(base)
            if origin is IModel:
                args = get_args(base)
                if args:
                    cfg = args[0]
                    if isinstance(cfg, type) and issubclass(cfg, BaseTrainConfig):
                        return cfg
        return None

    @staticmethod
    def register_model(name: str, model_type: Type[IModel[Any]]) -> None:
        ModelsFactory._types_dict[name] = model_type
        cfg_type = ModelsFactory._extract_config_type(model_type)
        if cfg_type:
            ModelsFactory._train_cfg_dict[name] = cfg_type

    @staticmethod
    def get_model_class(name: str) -> Type[IModel[Any]]:
        return ModelsFactory._types_dict[name]

    @staticmethod
    def get_config_class(name: str) -> Type[BaseTrainConfig]:
        return ModelsFactory._train_cfg_dict[name]

    @staticmethod
    def get_model_by_name(name: str, on_load_progress: OnLoadProgress) -> IModel[Any]:
        if name in ModelsFactory._models_dict:
            return ModelsFactory._models_dict[name]
        if (
            ModelsFactory.saver is None
            or ModelsFactory.db is None
            or ModelsFactory.dash is None
        ):
            raise RuntimeError("ModelsFactory.__init__(...) not performed")
        model_type = ModelsFactory._types_dict.get(name)
        if model_type is None:
            raise RuntimeError(f"Unknown model type: {name}")
        inst = model_type(
            ModelsFactory.saver,
            ModelsFactory.db,
            ModelsFactory.dash,
            name,
            on_load_progress,
        )
        ModelsFactory._models_dict[name] = inst
        return inst

    @staticmethod
    def get_names_list() -> List[str]:
        return list(ModelsFactory._types_dict.keys())

    @staticmethod
    def build_train_config_union():
        cfg_types = list(ModelsFactory._train_cfg_dict.values())
        if not cfg_types:
            return BaseTrainConfig
        union_type = Union[tuple(cfg_types)]
        return Annotated[union_type, Field(discriminator="model_name")]
