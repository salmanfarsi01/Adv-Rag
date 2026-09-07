from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .structured import LayoutBlock, StructuredDocument


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_num: int
    bbox: tuple[int, int, int, int]
    ocr_confidence: float
    block_types: tuple[str, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "page_num": self.page_num,
            "bbox": json.dumps(list(self.bbox), separators=(",", ":")),
            "ocr_confidence": self.ocr_confidence,
            "block_types": ",".join(self.block_types),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_bbox(blocks: list[LayoutBlock]) -> tuple[int, int, int, int]:
    return min(block.bbox[0] for block in blocks), min(block.bbox[1] for block in blocks), max(block.bbox[2] for block in blocks), max(block.bbox[3] for block in blocks)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def _make_chunk(doc_id: str, page_num: int, index: int, blocks: list[LayoutBlock]) -> Chunk:
    return Chunk(
        f"{doc_id}:p{page_num}:c{index}",
        doc_id,
        "\n".join(block.text for block in blocks),
        page_num,
        _merge_bbox(blocks),
        sum(block.confidence for block in blocks) / len(blocks),
        tuple(dict.fromkeys(block.type for block in blocks)),
    )


def chunk_document(document: StructuredDocument, max_chars: int = 1200) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0
    for page in document.pages:
        pending: list[LayoutBlock] = []
        pending_length = 0
        for block in page.blocks:
            is_table = block.type == "table_cell"
            if is_table:
                if pending:
                    chunks.append(_make_chunk(document.doc_id, page.page_num, chunk_index, pending))
                    chunk_index += 1
                    pending = []
                    pending_length = 0
                chunks.append(_make_chunk(document.doc_id, page.page_num, chunk_index, [block]))
                chunk_index += 1
                continue
            for sentence in _sentences(block.text) or [block.text]:
                if pending and pending_length + len(sentence) + 1 > max_chars:
                    chunks.append(_make_chunk(document.doc_id, page.page_num, chunk_index, pending))
                    chunk_index += 1
                    pending = []
                    pending_length = 0
                pending.append(LayoutBlock(sentence, block.bbox, block.confidence, block.type))
                pending_length += len(sentence) + 1
        if pending:
            chunks.append(_make_chunk(document.doc_id, page.page_num, chunk_index, pending))
    return chunks


def chunk_documents(documents: Iterable[StructuredDocument], max_chars: int = 1200) -> list[Chunk]:
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars))
    return chunks
