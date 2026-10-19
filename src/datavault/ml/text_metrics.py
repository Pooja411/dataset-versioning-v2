from typing import Any


def text_stats(texts: list[str]) -> dict[str, Any]:
    tokenised = [t.split() for t in texts]
    total = sum(len(t) for t in tokenised)
    vocab = {tok.lower() for toks in tokenised for tok in toks}
    return {
        "num_rows": len(texts),
        "vocabulary_size": len(vocab),
        "avg_sentence_len": round(total / len(texts), 4) if texts else 0.0,
        "total_tokens": total,
    }
