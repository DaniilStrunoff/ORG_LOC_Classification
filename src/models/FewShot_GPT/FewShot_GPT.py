import os, time, math, torch
from typing import Literal, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.Types import BaseTrainConfig, ModelPrediction, Label
from src.models import IModel, ModelsFactory
from src.controllers.HFModelController import HFModelController
from src.controllers.DBController import DBController
from src.controllers.DashboardController import DashboardController
from src.controllers.ModelSavingController import ModelSavingController
from src.controllers.ModelStateController import OnEpoch, OnLoadProgress

MODEL_NAME: str = "few_shot_gpt"


class FewShotGPTTrainConfig(BaseTrainConfig):
    model_name: Literal["few_shot_gpt"] = MODEL_NAME  # type: ignore


class FewShot_GPT(IModel[FewShotGPTTrainConfig]):
    BASE_MODEL = "google/gemma-2-2b-it"
    TINY_MODEL = "hf-internal-testing/tiny-random-Gemma2ForCausalLM"
    FEWSHOT = [
        ("The Dubliner", "ORG"),
        ("Тверская", "LOC"),
        ("Шоколадница", "ORG"),
        ("Пушкинская", "LOC"),
        ("Башня Федерации", "LOC"),
    ]
    INSTRUCTION = "\n".join(
        (
            "Ты — строгий классификатор.",
            "Отвечай ORG, если запрос про заведение, кафе, ресторан, компанию, бренд, кинотеатр, театр, сеть магазинов, продуктовый, банк и т.д.",
            "Отвечай LOC, если запрос про город, улицу, парк, площадь, район, памятник, собор, храм, именованное здание, озеро, реку или любое другое географическое место.",
            "Отвечай ровно одним словом: ORG или LOC.",
        )
    )
    DECISION_THRESHOLD = 0.5

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
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        hf = HFModelController()
        self.tokenizer = hf.load_model(
            self.BASE_MODEL,
            self.TINY_MODEL,
            model_class=AutoTokenizer,
            with_tokenizer=True,
            progress=on_load_progress,
            token=token,
        )
        self.model = hf.load_model(
            self.BASE_MODEL,
            self.TINY_MODEL,
            model_class=AutoModelForCausalLM,
            progress=on_load_progress,
            token=token,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.device = self.model.device
        self.ready = True

    def _build_prompt(self, name: str) -> str:
        shots = "\n".join([f"{a} → {b}" for a, b in self.FEWSHOT])
        return f"{self.INSTRUCTION}\nПримеры:\n{shots}\nКлассифицируй: {name}\nОтвет:"

    def _prompt_ids_for(self, name: str) -> torch.Tensor:
        prompt = self._build_prompt(name)
        enc = self.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return enc["input_ids"].to(self.device)

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def _score_candidate(self, prompt_ids: torch.Tensor, candidate: str) -> float:
        cand = self.tokenizer(candidate, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(self.device)
        x = torch.cat([prompt_ids, cand], dim=1)
        labels = x.clone()
        labels[:, : prompt_ids.shape[1]] = -100
        out = self.model(input_ids=x, labels=labels)
        cand_len = cand.shape[1]
        return -out.loss.item() * cand_len

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def _p_org_for(self, name: str) -> float:
        pids = self._prompt_ids_for(name)
        s_org = self._score_candidate(pids, "ORG")
        s_loc = self._score_candidate(pids, "LOC")
        m = max(s_org, s_loc)
        e_org, e_loc = math.exp(s_org - m), math.exp(s_loc - m)
        return e_org / (e_org + e_loc)

    def train(
        self,
        config: FewShotGPTTrainConfig,
        train_data: List[tuple[str, Label]],
        val_data: List[tuple[str, Label]],
        on_epoch: OnEpoch,
    ) -> None:
        on_epoch(0, 0, 0, 1, 1, 1)

    def _decide(
        self, p_org: float, t: float | None = None
    ) -> Literal["ORG", "LOC", "UNKNOWN"]:
        t = self.DECISION_THRESHOLD if t is None else float(t)
        return "ORG" if p_org >= t else "LOC"

    def predict_one(self, text: str, threshold: float | None = None) -> ModelPrediction:
        t0 = time.time()
        p_org = self._p_org_for(text)
        label = self._decide(p_org, t=threshold)
        confidence = p_org if label == "ORG" else 1.0 - p_org
        latency_ms = round((time.time() - t0) * 1000, 2)
        return ModelPrediction(
            text=text,
            gold="UNKNOWN",
            label=label,
            confidence=round(float(confidence), 4),
            latency_ms=latency_ms,
            error=False,
            hint="",
        )


ModelsFactory.register_model(MODEL_NAME, FewShot_GPT)
