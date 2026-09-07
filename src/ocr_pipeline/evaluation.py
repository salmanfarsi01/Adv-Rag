from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class EvaluationQuery:
    query: str
    relevant_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    query: str
    k: int
    retrieved_chunk_ids: tuple[str, ...]
    hit: bool


def precision_at_k(results: list[EvaluationResult]) -> float:
    return sum(result.hit for result in results) / len(results) if results else 0.0


def evaluate(queries: list[EvaluationQuery], search: Callable[[str, int], list[dict[str, Any]]], k: int) -> list[EvaluationResult]:
    results = []
    for item in queries:
        retrieved = search(item.query, k)
        ids = tuple(result["metadata"]["chunk_id"] for result in retrieved)
        results.append(EvaluationResult(item.query, k, ids, bool(set(ids) & set(item.relevant_chunk_ids))))
    return results


def load_queries(path: Path) -> list[EvaluationQuery]:
    with path.open(encoding="utf-8") as stream:
        return [EvaluationQuery(item["query"], tuple(item["relevant_chunk_ids"])) for item in json.load(stream)]


def write_evaluation(results: list[EvaluationResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"precision_at_k": precision_at_k(results), "results": [asdict(result) for result in results]}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
