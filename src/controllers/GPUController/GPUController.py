from dataclasses import dataclass, field
from typing import List, Optional
import os, math, json
from src.controllers.IController import IController


def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = min(int(math.log(n, 1024)), len(units) - 1)
    v = n / (1024**i)
    if units[i] in ("GB", "TB"):
        return f"{v:.1f} {units[i]}"
    return f"{int(v)} {units[i]}"


@dataclass(frozen=True, slots=True)
class GPUDevice:
    id: str
    name: str
    total_mem: str


@dataclass(frozen=True, slots=True)
class GPUReport:
    gpu_visible: bool
    devices: List[GPUDevice] = field(default_factory=lambda: [])
    via: Optional[str] = None
    error: Optional[str] = None


class GPUController(IController):
    def probe(self) -> GPUReport:
        via = "torch.cuda"
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                devs: List[GPUDevice] = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)  # type: ignore
                    mem = getattr(props, "total_memory", None)  # type: ignore
                    devs.append(
                        GPUDevice(
                            id=f"cuda:{i}",
                            name=str(props.name),  # type: ignore
                            total_mem=_fmt_bytes(int(mem)) if mem is not None else "—",
                        )
                    )
                return GPUReport(gpu_visible=True, via=via, devices=devs)

            hint_env = os.getenv("CUDA_VISIBLE_DEVICES")
            hint = "CUDA not available (torch.cuda.is_available() == False)."
            if hint_env is not None:
                hint += f" CUDA_VISIBLE_DEVICES={json.dumps(hint_env)}."
            return GPUReport(gpu_visible=False, via=via, devices=[], error=hint)

        except Exception as e:
            hint_env = os.getenv("CUDA_VISIBLE_DEVICES")
            hint = f"Failed to check CUDA: {e.__class__.__name__}: {e}"
            if hint_env is not None:
                hint += f" | CUDA_VISIBLE_DEVICES={json.dumps(hint_env)}"
            return GPUReport(gpu_visible=False, via=via, devices=[], error=hint)
