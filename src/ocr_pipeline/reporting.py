from __future__ import annotations

import csv
from pathlib import Path

from .models import OCRDocument


def write_accuracy_report(documents: list[OCRDocument], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for document in documents:
        for page in document.pages:
            rows.append({
                "source": document.source,
                "page": page.page_number,
                "engine": page.engine,
                "words": len(page.words),
                "mean_confidence": round(page.mean_confidence, 2),
                "low_confidence_ratio": round(page.low_confidence_ratio, 4),
                "characters": len(page.text),
                "error": page.error or "",
            })
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["source"])
        writer.writeheader()
        writer.writerows(rows)
