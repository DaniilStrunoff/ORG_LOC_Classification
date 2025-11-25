import os, time, math, torch
import torch.nn.functional as F
from typing import Optional, List, Literal
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from src.models import IModel, ModelsFactory
from src.models.Types import ModelPrediction, Label, BaseTrainConfig
from src.controllers.HFModelController import HFModelController
from src.controllers.DBController import DBController
from src.controllers.DashboardController import DashboardController
from src.controllers.ModelSavingController import ModelSavingController
from src.controllers.ModelStateController import OnEpoch, OnLoadProgress

MODEL_NAME: str = "few_shot_mlm"


class FewShotMLMTrainConfig(BaseTrainConfig):
    model_name: Literal["few_shot_mlm"] = MODEL_NAME  # type: ignore


class FewShot_MLM(IModel[FewShotMLMTrainConfig]):
    BASE_MODEL = "xlm-roberta-large"
    TINY_MODEL = "hf-internal-testing/tiny-xlm-roberta"
    FEWSHOT_TXT = [
        "«Шоколадница» — это кафе.",
        "«Тверская» — это улица.",
        "«Большой театр» — это театр.",
        "«Парк Горького» — это парк.",
        "«Башня Федерации» — это здание.",
    ]
    DECISION_THRESHOLD = 0.21

    def __init__(
        self,
        saver: Optional[ModelSavingController] = None,
        db: Optional[DBController] = None,
        dash: Optional[DashboardController] = None,
        model_name: str = MODEL_NAME,
        on_load_progress: OnLoadProgress | None = None,
    ) -> None:
        self.saver = saver or ModelSavingController()
        self.db = db or DBController()
        self.dash = dash or DashboardController()
        self.model_name = model_name

        hf = HFModelController()

        self.tokenizer = hf.load_model(
            self.BASE_MODEL,
            self.TINY_MODEL,
            model_class=AutoTokenizer,
            with_tokenizer=True,
            progress=on_load_progress,
            use_fast=True,
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        )
        assert isinstance(self.tokenizer, PreTrainedTokenizerBase)
        self.tokenizer.truncation_side = "left"
        if (
            self.tokenizer.pad_token is None
            and getattr(self.tokenizer, "eos_token", None) is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = hf.load_model(
            self.BASE_MODEL,
            self.TINY_MODEL,
            model_class=AutoModelForMaskedLM,
            progress=on_load_progress,
            device_map="auto",
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        )
        assert isinstance(self.model, PreTrainedModel)
        self.model.eval()

        self.device = self.model.device
        self.ready = True

        self.ORG_WORDS = [
            "компания",
            "организация",
            "бренд",
            "кафе",
            "ресторан",
            "бар",
            "банк",
            "театр",
            "магазин",
            "отель",
            "аптека",
            "супермаркет",
            "заведение",
        ]
        self.LOC_WORDS = [
            "город",
            "улица",
            "проспект",
            "набережная",
            "бульвар",
            "переулок",
            "площадь",
            "станция",
            "метро",
            "парк",
            "район",
            "деревня",
            "посёлок",
            "место",
            "здание",
        ]

        def words_to_ids_list(words: List[str]) -> List[List[int]]:
            out: List[List[int]] = []
            for w in words:
                ids = (
                    self.tokenizer(w, add_special_tokens=False, return_tensors="pt")
                    .input_ids[0]
                    .tolist()
                )
                if ids:
                    out.append(ids)
            return out

        self.vids = {
            "ORG": words_to_ids_list(self.ORG_WORDS),
            "LOC": words_to_ids_list(self.LOC_WORDS),
        }
        assert (
            self.vids["ORG"] and self.vids["LOC"]
        ), "Верифицируй вербалайзеры: пустые списки"
        self.K = max(
            max(len(ids) for ids in self.vids["ORG"]),
            max(len(ids) for ids in self.vids["LOC"]),
        )
        assert (
            self.tokenizer.mask_token_id is not None
        ), "Нужна MLM-модель с mask_token_id"

    def _make_prompt(self, text: str) -> str:
        masks = " ".join([str(self.tokenizer.mask_token)] * self.K)
        fewshot = "\n".join(self.FEWSHOT_TXT)
        return f"{fewshot}\n«{text}» — это {masks}."

    def _prompt_ids(self, text: str):
        prompt = self._make_prompt(text)
        max_pos = int(getattr(self.model.config, "max_position_embeddings", 512)) - 2
        enc = self.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_pos,
        )

        ids = enc.input_ids[0]
        att = enc.attention_mask[0]
        m_id = self.tokenizer.mask_token_id
        mask_pos = (ids == m_id).nonzero(as_tuple=False).flatten().tolist()
        if len(mask_pos) < self.K:
            ids[-self.K :] = m_id
            att[-self.K :] = 1
            mask_pos = list(range(ids.numel() - self.K, ids.numel()))
        return (
            ids.unsqueeze(0).to(self.device),
            att.unsqueeze(0).to(self.device),
            mask_pos,
        )

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def _score_variants(
        self, logits: torch.Tensor, mask_pos: List[int], variants: List[List[int]]
    ) -> float:
        logp = F.log_softmax(logits, dim=-1)[0]
        best: Optional[float] = None
        for token_ids in variants:
            L = min(len(token_ids), len(mask_pos))
            if L == 0:
                continue
            s = 0.0
            for j in range(L):
                s += logp[mask_pos[j], token_ids[j]].item()
            best = s if best is None else max(best, s)
        return best if best is not None else -1e9

    def _p_org_for(self, text: str) -> float:
        x, a, mask_pos = self._prompt_ids(text)
        with torch.no_grad():
            out = self.model(input_ids=x, attention_mask=a)
            logits = out.logits
        s_org = self._score_variants(logits, mask_pos, self.vids["ORG"])
        s_loc = self._score_variants(logits, mask_pos, self.vids["LOC"])
        m = max(s_org, s_loc)
        e_org, e_loc = math.exp(s_org - m), math.exp(s_loc - m)
        return e_org / (e_org + e_loc)

    def _decide(self, p_org: float, t: float | None = None) -> Label:
        t = self.DECISION_THRESHOLD if t is None else float(t)
        return "ORG" if p_org >= t else "LOC"

    def train(
        self,
        config: FewShotMLMTrainConfig,
        train_data: List[tuple[str, Label]],
        val_data: List[tuple[str, Label]],
        on_epoch: OnEpoch,
    ) -> None:
        on_epoch(0, 0, 0, 1, 1, 1)

    def predict_one(self, text: str, threshold: float | None = None) -> ModelPrediction:
        t0 = time.time()
        p_org = float(self._p_org_for(text))
        label = self._decide(p_org, t=threshold)
        confidence = p_org if label == "ORG" else 1.0 - p_org
        latency_ms = round((time.time() - t0) * 1000, 2)
        return ModelPrediction(
            text=text,
            gold="UNKNOWN",
            label=label,
            confidence=round(confidence, 4),
            latency_ms=latency_ms,
            error=False,
            hint="",
        )


ModelsFactory.register_model(MODEL_NAME, FewShot_MLM)
