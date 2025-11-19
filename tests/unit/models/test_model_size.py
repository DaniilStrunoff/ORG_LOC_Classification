from typing import Iterable, Mapping, List
import pytest
from torch import nn
from src.models import ModelsFactory


def find_all_torch_modules(obj: object) -> List[nn.Module]:
    seen_objs: set[int] = set()
    found: dict[int, nn.Module] = {}

    def walk(o: object) -> None:
        if isinstance(o, nn.Module):
            mid = id(o)
            if mid not in found:
                found[mid] = o
            return

        oid = id(o)
        if oid in seen_objs:
            return
        seen_objs.add(oid)

        if isinstance(o, Mapping):
            for v in o:  # type: ignore
                if not isinstance(v, object):
                    continue
                walk(v)
            return

        if isinstance(o, Iterable) and not isinstance(o, (str, bytes, bytearray)):
            for v in o:  # type: ignore
                if not isinstance(v, object):
                    continue
                walk(v)
            return

        try:
            values = vars(o).values()
        except TypeError:
            return

        for v in values:
            walk(v)

    walk(obj)
    if not found:
        raise TypeError(f"Cannot find torch.nn.Module inside {type(obj)}")
    return list(found.values())


def get_model_param_size_bytes_any(obj: object) -> int:
    modules = find_all_torch_modules(obj)
    seen_tensors: set[int] = set()
    total = 0
    for module in modules:
        for t in module.state_dict().values():
            if getattr(t, "device", None) is not None and t.device.type == "meta":
                continue
            tid = id(t)
            if tid in seen_tensors:
                continue
            seen_tensors.add(tid)
            total += t.numel() * t.element_size()
    return total


@pytest.mark.parametrize("model_name", ModelsFactory.get_names_list())
def test_model_size_is_small(model_name: str):
    model_cls = ModelsFactory.get_model_class(model_name)
    model = model_cls()

    size = get_model_param_size_bytes_any(model)

    print(f"\nMODEL {model_name} size: {size/1024:.2f} KB")

    assert size < 100 * 1024 * 1024, f"Model {model_name} is too large: {size} bytes"
