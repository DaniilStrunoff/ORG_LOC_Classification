from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Protocol, TYPE_CHECKING

import requests
from huggingface_hub import HfApi, hf_hub_url
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from src.controllers.IController import IController

if TYPE_CHECKING:
    from src.controllers.ModelStateController import OnLoadProgress


def _cache_root() -> str:
    return os.getenv("HF_HUB_CACHE", "/artifacts/hf_cache")


def _ensure_dir(p: Path | str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _wanted_files(files: List[str], need_tokenizer: bool) -> List[str]:
    wanted: List[str] = []
    if "config.json" in files:
        wanted.append("config.json")
    if any(f.endswith(".safetensors") for f in files):
        wanted += [f for f in files if f.endswith(".safetensors")]
        wanted += [
            f
            for f in files
            if f.endswith("safetensors.index.json")
            or f.endswith("model.safetensors.index.json")
        ]
    else:
        wanted += [
            f
            for f in files
            if f == "pytorch_model.bin" or f.startswith("pytorch_model-")
        ]
        wanted += [f for f in files if f.endswith("pytorch_model.bin.index.json")]
    if need_tokenizer:
        tok_exact = {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "spm.model",
        }
        wanted += [f for f in files if f in tok_exact]
        for f in files:
            fn = f.lower()
            if fn.endswith(".model") and (
                "sentencepiece" in fn or "spm" in fn or "ice_text" in fn
            ):
                wanted.append(f)
            if fn.endswith((".vocab", ".spm")):
                wanted.append(f)
    return sorted(set(wanted))


def _stream_file(
    url: str,
    dst: Path | str,
    headers: Dict[str, str],
    agg: Dict[str, int],
    cb: OnLoadProgress | None,
    chunk: int = 1024 * 1024,
) -> None:
    dst = Path(dst)
    tmp = dst.with_suffix(dst.suffix + ".incomplete")
    _ensure_dir(dst.parent)
    start = 0
    if dst.exists():
        start = dst.stat().st_size
        dst.replace(tmp)
    elif tmp.exists():
        start = tmp.stat().st_size
    hdrs = dict(headers)
    if start > 0:
        hdrs["Range"] = f"bytes={start}-"
    with requests.get(url, headers=hdrs, stream=True) as r:
        r.raise_for_status()
        with tmp.open("ab") as f:
            for chunk_bytes in r.iter_content(chunk_size=chunk):
                if not chunk_bytes:
                    continue
                f.write(chunk_bytes)
                agg["done"] += len(chunk_bytes)
                if cb:
                    total = (
                        agg["total"]
                        if (agg["total"] and agg["total"] >= agg["done"])
                        else agg["done"]
                    )
                    cb(agg["done"], total)
    tmp.replace(dst)


class ModelFactory(Protocol):
    @classmethod
    def from_pretrained(cls, repo_id: str, /, **kwargs: Any) -> PreTrainedModel: ...


class TokenizerFactory(Protocol):
    @classmethod
    def from_pretrained(
        cls, repo_id: str, /, **kwargs: Any
    ) -> PreTrainedTokenizerBase: ...


class HFModelController(IController):
    def __init__(self) -> None:
        self.cache = _cache_root()
        _ensure_dir(self.cache)

    def load_model(
        self,
        repo_id: str,
        tiny_repo_id: str,
        model_class: type[ModelFactory] | type[TokenizerFactory] = AutoModel,
        revision: str = "main",
        token: str | None = None,
        with_tokenizer: bool = False,
        progress: OnLoadProgress | None = None,
        **from_pretrained_kwargs: Any,
    ) -> PreTrainedModel | PreTrainedTokenizerBase:
        api = HfApi()
        info = api.model_info(
            repo_id=repo_id, revision=revision, token=token, files_metadata=True
        )

        if model_class is AutoTokenizer:
            with_tokenizer = True

        siblings = info.siblings or []
        files = [s.rfilename for s in siblings] if siblings else []
        wanted = _wanted_files(files, with_tokenizer)

        rid = repo_id.replace("/", "--")
        sha = info.sha or ""
        snap_dir = Path(self.cache) / f"models--{rid}" / "snapshots" / sha
        _ensure_dir(snap_dir)

        size_by_name: dict[str, int] = {}
        for s in siblings:
            size = getattr(s, "size", None)
            if size is not None:
                size_by_name[s.rfilename] = int(size)

        headers = {"Authorization": f"Bearer {token}"} if token else {}

        LARGE = 10**12
        sizes: dict[str, int] = {}
        for f in wanted:
            size = size_by_name.get(f)
            if size is None:
                size = LARGE
            sizes[f] = size

        cached = 0
        for f in wanted:
            dst = snap_dir / f
            if dst.exists():
                try:
                    s = dst.stat().st_size
                    cached += min(s, sizes[f])
                except OSError:
                    pass

        agg: dict[str, int] = {"done": cached, "total": sum(sizes.values(), 0)}

        for f in wanted:
            dst = snap_dir / f
            need = sizes[f]
            if dst.exists() and dst.stat().st_size >= need:
                if progress:
                    total = agg["total"] if agg["total"] >= agg["done"] else agg["done"]
                    progress(agg["done"], total)
                continue
            url = hf_hub_url(repo_id=repo_id, filename=f, revision=revision)
            _stream_file(url, dst, headers, agg, progress)

        if progress:
            total = agg["total"]
            progress(int(total), int(total))

        kwargs: Dict[str, Any] = {"local_files_only": True, **from_pretrained_kwargs}
        return model_class.from_pretrained(str(snap_dir), **kwargs)
