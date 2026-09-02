"""Map noisy Shenava transcripts onto catalog service names."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from src.utils import normalize_persian

_VOWELS = set("اآأإویىئهة")


def fold_fa(text: str) -> str:
    t = normalize_persian(text)
    t = t.replace("‌", "").replace("-", "").replace(" ", "")
    t = t.replace("آ", "ا").replace("أ", "ا").replace("ة", "ه")
    return t


def skeleton_fa(text: str) -> str:
    return "".join(ch for ch in fold_fa(text) if ch not in _VOWELS)


def _pair_score(utterance: str, keyword: str) -> float:
    u, k = fold_fa(utterance), fold_fa(keyword)
    if not u or not k or len(k) < 3:
        return 0.0
    if k in u or (len(u) >= 4 and u in k):
        return 1.0
    words = [
        fold_fa(w)
        for w in normalize_persian(utterance).split()
        if len(fold_fa(w)) >= 3
    ]
    candidates = list(dict.fromkeys(words + ([u] if len(u) >= 4 else [])))
    best = 0.0
    sk = skeleton_fa(keyword)
    for cu in candidates:
        if k in cu or (len(cu) >= 4 and cu in k):
            return 1.0
        if abs(len(cu) - len(k)) <= 2:
            fold_r = SequenceMatcher(None, cu, k).ratio()
            if fold_r >= 0.88:
                best = max(best, fold_r)
        scu = skeleton_fa(cu)
        if len(scu) < 3 or len(sk) < 3:
            continue
        if min(len(scu), len(sk)) / max(len(scu), len(sk)) < 0.7:
            continue
        ratio = SequenceMatcher(None, scu, sk).ratio()
        need = 0.9 if max(len(scu), len(sk)) >= 5 else 0.82
        if ratio >= need:
            best = max(best, ratio)
    return best


def score_service(text: str, svc: dict[str, Any]) -> float:
    names = [svc.get("name") or ""] + (svc.get("keywords") or "").split()
    return max((_pair_score(text, n) for n in names if n), default=0.0)


def best_service(text: str, services: list[dict[str, Any]], min_score: float = 0.78):
    """Return (service, score) when the transcript is close enough to a catalog item."""
    t = normalize_persian(text)
    if not t or not services:
        return None, 0.0
    ranked = [(score_service(t, svc), svc) for svc in services]
    ranked.sort(key=lambda x: x[0], reverse=True)
    score, svc = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    if score >= min_score and score >= second + 0.04:
        return svc, score
    return None, score


def snap_heard_text(text: str, services: list[dict[str, Any]]) -> str:
    """Keep exact phrases; replace fuzzy-only hits with the canonical service name."""
    t = normalize_persian(text)
    svc, score = best_service(t, services)
    if not svc:
        return t
    name = svc["name"]
    if name in t:
        return t
    for kw in (svc.get("keywords") or "").split():
        if len(kw) >= 3 and kw in t:
            return t
    if score >= 0.78:
        return name
    return t
