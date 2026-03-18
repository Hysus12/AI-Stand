from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def simple_tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def infer_question_topic(question_text: str) -> str:
    lowered = question_text.lower()
    if any(token in lowered for token in ["government", "democracy", "election", "vote"]):
        return "politics"
    if any(token in lowered for token in ["religion", "church", "god"]):
        return "religion"
    if any(token in lowered for token in ["work", "job", "income", "economy"]):
        return "economy"
    if any(token in lowered for token in ["family", "marriage", "children"]):
        return "family"
    return "general"
