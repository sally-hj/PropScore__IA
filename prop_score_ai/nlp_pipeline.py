"""CamemBERT semantic analysis with keyword and embedding fallback."""

from __future__ import annotations

import os
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional import guard
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from .constants import (
    DEFAULT_CAMEMBERT_MODEL,
    HESITATION_KEYWORDS,
    LABELS,
    NEGATIVE_KEYWORDS,
    POSITIVE_KEYWORDS,
)
from .io_utils import dataframe_to_csv
from .logging_utils import setup_logger

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - optional import guard
    AutoModel = None
    AutoTokenizer = None

try:
    import torch
except Exception:  # pragma: no cover - optional import guard
    torch = None


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _count_keywords(text: str, keywords: list[str]) -> int:
    if not text:
        return 0
    score = 0
    for kw in keywords:
        if " " in kw:
            score += text.count(kw.lower())
        else:
            score += len(re.findall(rf"\b{re.escape(kw.lower())}\b", text))
    return score


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    mean_pooled = summed / counts
    return mean_pooled.squeeze(0).detach().cpu().numpy()


@dataclass
class SemanticResult:
    sentiment_label: str
    sentiment_score: float
    keyword_count: int
    positive_keyword_count: int
    hesitation_keyword_count: int
    negative_keyword_count: int
    keyword_type: str
    semantic_score: float
    analysis_method: str


class CamembertSemanticAnalyzer:
    """CamemBERT-based semantic analyzer with explicit fallback logic."""

    def __init__(self, model_name: str = DEFAULT_CAMEMBERT_MODEL, device: str | None = None):
        self.model_name = model_name
        if device is not None:
            self.device = device
        elif torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self._tokenizer = None
        self._model = None
        self._prototype_cache: dict[str, np.ndarray] = {}

    def _ensure_model(self):
        if torch is None:
            raise RuntimeError("torch is not available")
        if self._tokenizer is None or self._model is None:
            if AutoTokenizer is None or AutoModel is None:
                raise RuntimeError("transformers is not available")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
        return self._tokenizer, self._model

    def _embed(self, text: str) -> np.ndarray:
        if torch is None:
            raise RuntimeError("torch is not available")
        tokenizer, model = self._ensure_model()
        encoded = tokenizer(text or "", return_tensors="pt", truncation=True, max_length=128, padding=True)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        return _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])

    def _prototype_embedding(self, name: str, text: str) -> np.ndarray:
        if name not in self._prototype_cache:
            self._prototype_cache[name] = self._embed(text)
        return self._prototype_cache[name]

    def analyze(self, transcript: str) -> dict:
        text = _normalize_text(transcript or "")
        pos = _count_keywords(text, POSITIVE_KEYWORDS)
        hes = _count_keywords(text, HESITATION_KEYWORDS)
        neg = _count_keywords(text, NEGATIVE_KEYWORDS)
        kw_total = pos + hes + neg

        if not text:
            return SemanticResult(
                sentiment_label="neutral",
                sentiment_score=0.0,
                keyword_count=0,
                positive_keyword_count=0,
                hesitation_keyword_count=0,
                negative_keyword_count=0,
                keyword_type="neutral",
                semantic_score=0.0,
                analysis_method="empty_transcript",
            ).__dict__

        try:
            self._ensure_model()
            emb = self._embed(text)
            interest_proto = self._prototype_embedding("interest", "je suis très intéressé par ce bien immobilier")
            hes_proto = self._prototype_embedding("hesitation", "je réfléchis encore et je veux comparer plusieurs biens")
            refusal_proto = self._prototype_embedding("refusal", "je ne suis pas intéressé je refuse et je ne donnerai pas suite")

            sim_interest = _cosine_similarity(emb, interest_proto)
            sim_hesitation = _cosine_similarity(emb, hes_proto)
            sim_refusal = _cosine_similarity(emb, refusal_proto)

            interest_score = pos * 1.5 + max(sim_interest, 0.0) * 2.0
            hesitation_score = hes * 1.5 + max(sim_hesitation, 0.0) * 1.5
            refusal_score = neg * 1.5 + max(sim_refusal, 0.0) * 2.0
            scores = {
                "interest": interest_score,
                "hesitation": hesitation_score,
                "refusal": refusal_score,
            }
            keyword_type = max(scores, key=scores.get) if any(scores.values()) else "neutral"
            sentiment_label = keyword_type
            max_score = max(scores.values()) if any(scores.values()) else 0.0
            total = sum(scores.values()) or 1.0
            sentiment_score = float(np.clip((interest_score - refusal_score) / total, -1.0, 1.0))
            semantic_score = float(np.clip(50.0 + 25.0 * sentiment_score + 10.0 * (hesitation_score / total), 0.0, 100.0))
            return SemanticResult(
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score,
                keyword_count=kw_total,
                positive_keyword_count=pos,
                hesitation_keyword_count=hes,
                negative_keyword_count=neg,
                keyword_type=keyword_type,
                semantic_score=semantic_score,
                analysis_method="camembert_embeddings_keyword",
            ).__dict__
        except Exception:
            return self._keyword_fallback(text, pos, hes, neg)

    def _keyword_fallback(self, text: str, pos: int, hes: int, neg: int) -> dict:
        kw_total = pos + hes + neg
        if kw_total == 0:
            keyword_type = "neutral"
            semantic_score = 45.0
            sentiment_score = 0.0
        elif pos >= max(hes, neg):
            keyword_type = "interest"
            semantic_score = float(np.clip(65.0 + 6.0 * pos - 3.0 * neg + 2.0 * hes, 0.0, 100.0))
            sentiment_score = float(np.clip(0.6 + 0.08 * pos - 0.04 * neg, -1.0, 1.0))
        elif hes >= max(pos, neg):
            keyword_type = "hesitation"
            semantic_score = float(np.clip(50.0 + 3.0 * hes - 2.0 * neg, 0.0, 100.0))
            sentiment_score = float(np.clip(0.1 + 0.05 * hes - 0.03 * neg, -1.0, 1.0))
        else:
            keyword_type = "refusal"
            semantic_score = float(np.clip(25.0 - 6.0 * neg + 2.0 * hes, 0.0, 100.0))
            sentiment_score = float(np.clip(-0.6 - 0.08 * neg + 0.03 * hes, -1.0, 1.0))

        return SemanticResult(
            sentiment_label=keyword_type,
            sentiment_score=sentiment_score,
            keyword_count=kw_total,
            positive_keyword_count=pos,
            hesitation_keyword_count=hes,
            negative_keyword_count=neg,
            keyword_type=keyword_type,
            semantic_score=semantic_score,
            analysis_method="keyword_rule_fallback",
        ).__dict__


def process_semantic_dataset(
    transcripts_csv: str | Path,
    output_csv: str | Path,
    model_name: str = DEFAULT_CAMEMBERT_MODEL,
    logger=None,
) -> pd.DataFrame:
    logger = logger or setup_logger("camembert_nlp")
    transcripts_csv = Path(transcripts_csv)
    if not transcripts_csv.exists():
        df = pd.DataFrame()
        dataframe_to_csv(df, output_csv)
        return df

    transcripts = pd.read_csv(transcripts_csv)
    if transcripts.empty:
        dataframe_to_csv(transcripts, output_csv)
        return transcripts

    analyzer = CamembertSemanticAnalyzer(model_name=model_name)
    rows: list[dict] = []
    for _, row in tqdm(transcripts.iterrows(), total=len(transcripts), desc="CamemBERT"):
        analysis = analyzer.analyze(str(row.get("transcript", "")))
        analysis.update(
            {
                "prospect_id": row.get("prospect_id"),
                "video_id": row.get("video_id"),
                "label": row.get("label"),
                "transcript": row.get("transcript", ""),
                "language": row.get("language", "unknown"),
                "duration": row.get("duration", 0.0),
            }
        )
        rows.append(analysis)
    df = pd.DataFrame(rows)
    dataframe_to_csv(df, output_csv)
    logger.info("Saved semantic features to %s", output_csv)
    return df
