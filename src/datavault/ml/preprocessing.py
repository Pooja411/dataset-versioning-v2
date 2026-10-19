import string
from typing import Any

from ..errors import ValidationFailed

STEPS = (
    "strip_whitespace",
    "lowercase",
    "remove_punctuation",
    "remove_duplicates",
    "tokenize",
    "remove_stopwords",
)

STOPWORDS = frozenset(
    """a an the and or but if in on at to for of with by from is was are were be been being
    have has had do does did will would could should may might shall can need that this these
    those it its i me my we our you your he his she her they their what which who whom when
    where why how not no nor so yet both either neither as than then just because while
    although about above after before between into through during such up out over also more
    most other some any each there here all""".split()
)

_PUNCT = str.maketrans("", "", string.punctuation)

Row = tuple[str, str | None]


def normalize_config(config: dict[str, Any]) -> dict[str, bool]:
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    unknown = sorted(set(clean) - set(STEPS))
    if unknown:
        raise ValidationFailed(f"unknown preprocessing steps: {unknown}")
    bad = sorted(k for k, v in clean.items() if not isinstance(v, bool))
    if bad:
        raise ValidationFailed(f"preprocessing steps must be booleans: {bad}")
    return {step: clean.get(step, False) for step in STEPS}


def split_label(line: str) -> Row:
    cut = line.rfind("|")
    if cut < 0:
        return line.strip(), None
    text, label = line[:cut].strip(), line[cut + 1 :].strip()
    if text and label:
        return text, label
    return line.strip(), None


def _early(text: str, cfg: dict[str, bool]) -> str:
    if cfg["strip_whitespace"]:
        text = " ".join(text.split())
    if cfg["lowercase"]:
        text = text.lower()
    if cfg["remove_punctuation"]:
        text = text.translate(_PUNCT)
    return text


def _late(text: str, cfg: dict[str, bool]) -> str:
    if cfg["tokenize"]:
        text = " ".join(text.split())
    if cfg["remove_stopwords"]:
        text = " ".join(t for t in text.split() if t.lower() not in STOPWORDS)
    return text


def preprocess(rows: list[Row], config: dict[str, Any]) -> list[Row]:
    cfg = normalize_config(config)
    # duplicates must be judged on normalised text and label together, across all rows
    staged = [(_early(text, cfg), label) for text, label in rows]
    if cfg["remove_duplicates"]:
        seen: set[Row] = set()
        unique = []
        for row in staged:
            if row not in seen:
                seen.add(row)
                unique.append(row)
        staged = unique
    done = [(_late(text, cfg), label) for text, label in staged]
    return [(text, label) for text, label in done if text.strip()]


def preprocess_labeled_text(raw: bytes, config: dict[str, Any]) -> bytes:
    lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    rows = preprocess([split_label(ln) for ln in lines], config)
    out = [f"{text}|{label}" if label is not None else text for text, label in rows]
    return ("\n".join(out) + ("\n" if out else "")).encode("utf-8")
