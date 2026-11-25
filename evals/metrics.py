import math


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(
        1 / math.log2(i + 2) for i, d in enumerate(ranked_ids[:k]) if d in relevant
    )
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int = 10) -> float:
    return len(set(ranked_ids[:k]) & relevant) / len(relevant) if relevant else 0.0
