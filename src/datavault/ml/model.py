from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from ..errors import ValidationFailed

Vector = dict[int, float]


class TFIDFVectorizer:
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def fit(self, docs: list[str]) -> "TFIDFVectorizer":
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(set(doc.lower().split()))
        self.vocabulary = {tok: i for i, tok in enumerate(sorted(df))}
        n = len(docs)
        self.idf = {tok: math.log((1 + n) / (1 + count)) + 1.0 for tok, count in df.items()}
        return self

    def transform(self, docs: list[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for doc in docs:
            tokens = doc.lower().split()
            vec: Vector = {}
            for tok, count in Counter(tokens).items():
                if tok in self.vocabulary:
                    vec[self.vocabulary[tok]] = count / len(tokens) * self.idf[tok]
            vectors.append(vec)
        return vectors

    def fit_transform(self, docs: list[str]) -> list[Vector]:
        return self.fit(docs).transform(docs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def to_dict(self) -> dict[str, Any]:
        return {"vocabulary": self.vocabulary, "idf": self.idf}


class NaiveBayes:
    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValidationFailed("alpha must be positive")
        self.alpha = alpha
        self.classes: list[str] = []
        self.log_priors: dict[str, float] = {}
        self.log_probs: dict[str, list[float]] = {}

    def fit(self, vectors: list[Vector], labels: list[str], vocab_size: int) -> "NaiveBayes":
        self.classes = sorted(set(labels))
        counts = Counter(labels)
        self.log_priors = {c: math.log(counts[c] / len(labels)) for c in self.classes}
        weights: dict[str, Counter[int]] = {c: Counter() for c in self.classes}
        for vec, label in zip(vectors, labels):
            weights[label].update(vec)
        self.log_probs = {}
        for c in self.classes:
            denom = sum(weights[c].values()) + self.alpha * vocab_size
            self.log_probs[c] = [
                math.log((weights[c].get(i, 0.0) + self.alpha) / denom) for i in range(vocab_size)
            ]
        return self

    def scores(self, vec: Vector) -> dict[str, float]:
        return {
            c: self.log_priors[c] + sum(w * self.log_probs[c][i] for i, w in vec.items())
            for c in self.classes
        }

    def predict(self, vectors: list[Vector]) -> list[str]:
        out = []
        for vec in vectors:
            scored = self.scores(vec)
            out.append(max(self.classes, key=lambda c: (scored[c], c)))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "classes": self.classes,
            "log_priors": self.log_priors,
            "log_probs": self.log_probs,
        }


def stratified_split(
    texts: list[str], labels: list[str], test_ratio: float, seed: int
) -> tuple[list[str], list[str], list[str], list[str]]:
    if not 0 < test_ratio < 1:
        raise ValidationFailed("test_ratio must be between 0 and 1")
    by_class: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        by_class.setdefault(label, []).append(i)
    too_small = sorted(c for c, idx in by_class.items() if len(idx) < 2)
    if too_small:
        raise ValidationFailed(f"classes with fewer than 2 rows cannot be split: {too_small}")
    rng = random.Random(seed)
    train: list[int] = []
    test: list[int] = []
    for label in sorted(by_class):
        idx = list(by_class[label])
        rng.shuffle(idx)
        n_test = min(max(1, int(len(idx) * test_ratio)), len(idx) - 1)
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    train.sort()
    test.sort()
    pick = lambda xs, ids: [xs[i] for i in ids]  # noqa: E731
    return pick(texts, train), pick(texts, test), pick(labels, train), pick(labels, test)


def evaluate(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict[str, Any]:
    n = len(y_true)
    matrix = {t: {p: 0 for p in classes} for t in classes}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    per_class: dict[str, dict[str, float]] = {}
    f1_total = 0.0
    for c in classes:
        tp = matrix[c][c]
        fp = sum(matrix[t][c] for t in classes if t != c)
        fn = sum(matrix[c][p] for p in classes if p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_total += f1
        per_class[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    correct = sum(matrix[c][c] for c in classes)
    return {
        "accuracy": round(correct / n, 4) if n else 0.0,
        "macro_f1": round(f1_total / len(classes), 4),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "n": n,
    }


def top_features(
    clf: NaiveBayes, vec: TFIDFVectorizer, label: str, n: int = 8
) -> list[dict[str, Any]]:
    others = [c for c in clf.classes if c != label]
    tokens = sorted(vec.vocabulary, key=vec.vocabulary.get)  # type: ignore[arg-type]
    scored = []
    for idx, token in enumerate(tokens):
        baseline = sum(clf.log_probs[c][idx] for c in others) / len(others) if others else 0.0
        scored.append((token, clf.log_probs[label][idx] - baseline))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [{"token": t, "score": round(s, 4)} for t, s in scored[:n]]
