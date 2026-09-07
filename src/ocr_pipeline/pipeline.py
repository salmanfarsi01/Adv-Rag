from __future__ import annotations

import json
from pathlib import Path

import fitz

from .engines import FallbackOCR
from .models import OCRDocument, OCRPage


class PDFOCRPipeline:
    def __init__(self, ocr: FallbackOCR, dpi: int = 200) -> None:
        self.ocr = ocr
        self.dpi = dpi

    def process(self, path: Path) -> OCRDocument:
        document = fitz.open(path)
        pages: list[OCRPage] = []
        scale = self.dpi / 72
        matrix = fitz.Matrix(scale, scale)
        try:
            for page_index, pdf_page in enumerate(document):
                pixmap = pdf_page.get_pixmap(matrix=matrix, alpha=False)
                from PIL import Image
                image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                engine, result = self.ocr.extract(image)
                pages.append(OCRPage(page_index + 1, pixmap.width, pixmap.height, result.text, result.words, engine, result.mean_confidence, result.low_confidence_ratio))
        finally:
            document.close()
        return OCRDocument(str(path), pages)


def write_jsonl(documents: list[OCRDocument], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for document in documents:
            for page in document.pages:
                stream.write(json.dumps({"source": document.source, **page.to_dict()}, ensure_ascii=False) + "\n")
