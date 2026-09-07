from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import chunk_documents
from .evaluation import evaluate, load_queries, precision_at_k, write_evaluation
from .retrieval import LocalEmbedder, QdrantIndex
from .structured import structured_from_jsonl, write_structured_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query a Qdrant OCR vector index")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Create structured chunks and persist embeddings")
    build.add_argument("ocr_jsonl", type=Path)
    build.add_argument("--structured-output", type=Path, default=Path("data/output/structured.json"))
    build.add_argument("--index", type=Path, default=Path("data/qdrant"))
    build.add_argument("--url", help="Qdrant server URL; omit to use local persistent storage")
    build.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    build.add_argument("--max-chars", type=int, default=1200)
    build.add_argument("--batch-size", type=int, default=32)

    search = subparsers.add_parser("search", help="Search the persisted vector index")
    search.add_argument("query")
    search.add_argument("--index", type=Path, default=Path("data/qdrant"))
    search.add_argument("--url", help="Qdrant server URL; omit to use local persistent storage")
    search.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    search.add_argument("-k", type=int, default=5)

    score = subparsers.add_parser("evaluate", help="Calculate hit-based precision@k")
    score.add_argument("queries", type=Path)
    score.add_argument("--index", type=Path, default=Path("data/qdrant"))
    score.add_argument("--url", help="Qdrant server URL; omit to use local persistent storage")
    score.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    score.add_argument("--output", type=Path, default=Path("reports/retrieval_eval.json"))
    score.add_argument("-k", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build":
        documents = structured_from_jsonl(args.ocr_jsonl)
        write_structured_json(documents, args.structured_output)
        chunks = chunk_documents(documents, args.max_chars)
        index = QdrantIndex(args.index, embedder=LocalEmbedder(args.model), url=args.url)
        index.add(chunks, args.batch_size)
        print(json.dumps({"documents": len(documents), "chunks": len(chunks), "index": str(args.index)}))
        return

    index = QdrantIndex(args.index, embedder=LocalEmbedder(args.model), url=args.url)
    if args.command == "search":
        print(json.dumps(index.search(args.query, args.k), indent=2, ensure_ascii=False))
        return

    queries = load_queries(args.queries)
    results = evaluate(queries, index.search, args.k)
    write_evaluation(results, args.output)
    print(json.dumps({"queries": len(results), "k": args.k, "precision_at_k": precision_at_k(results)}))


if __name__ == "__main__":
    main()
