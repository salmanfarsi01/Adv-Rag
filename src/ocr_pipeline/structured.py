from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class LayoutBlock:
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float
    type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredPage:
    page_num: int
    blocks: list[LayoutBlock]

    def to_dict(self) -> dict[str, Any]:
        return {"page_num": self.page_num, "blocks": [block.to_dict() for block in self.blocks]}


@dataclass
class StructuredDocument:
    doc_id: str
    pages: list[StructuredPage]

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "pages": [page.to_dict() for page in self.pages]}


def _bbox(words: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    boxes = [word["bbox"] for word in words]
    return min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes)


def _group_lines(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    lines: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        if not lines or abs(word["bbox"][1] - lines[-1][0]["bbox"][1]) > max(8, word["bbox"][3] - word["bbox"][1]):
            lines.append([word])
        else:
            lines[-1].append(word)
    return [sorted(line, key=lambda item: item["bbox"][0]) for line in lines]


def _is_table_line(line: list[dict[str, Any]]) -> bool:
    if len(line) < 3:
        return False
    gaps = [right["bbox"][0] - left["bbox"][2] for left, right in zip(line, line[1:])]
    return sum(gap > 24 for gap in gaps) >= 2


def page_from_ocr(record: dict[str, Any]) -> StructuredPage:
    words = [word for word in record.get("words", []) if word.get("text", "").strip()]
    blocks = []
    for line in _group_lines(words):
        confidence = sum(float(word.get("confidence", 0)) for word in line) / len(line)
        blocks.append(LayoutBlock(" ".join(word["text"] for word in line), _bbox(line), confidence, "table_cell" if _is_table_line(line) else "paragraph"))
    return StructuredPage(int(record["page_number"]), blocks)


def structured_from_jsonl(path: Path, doc_id: str | None = None) -> list[StructuredDocument]:
    documents: dict[str, StructuredDocument] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            source = str(record.get("source", "document"))
            identifier = doc_id or Path(source).stem or re.sub(r"\W+", "_", source)
            document = documents.setdefault(identifier, StructuredDocument(identifier, []))
            document.pages.append(page_from_ocr(record))
    return list(documents.values())


def write_structured_json(documents: Iterable[StructuredDocument], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([document.to_dict() for document in documents], indent=2, ensure_ascii=False), encoding="utf-8")
