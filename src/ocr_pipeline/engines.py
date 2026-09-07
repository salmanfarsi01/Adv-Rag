from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .models import OCRWord


@dataclass
class EngineResult:
    text: str
    words: list[OCRWord]
    mean_confidence: float
    low_confidence_ratio: float


class OCREngine(ABC):
    name: str

    @abstractmethod
    def extract(self, image: Image.Image) -> EngineResult:
        raise NotImplementedError


def _metrics(words: list[OCRWord]) -> tuple[float, float]:
    if not words:
        return 0.0, 1.0
    confidences = [word.confidence for word in words]
    return sum(confidences) / len(confidences), sum(value < 60 for value in confidences) / len(confidences)


def _lines_from_words(words: list[OCRWord]) -> str:
    if not words:
        return ""
    ordered = sorted(words, key=lambda word: (word.bbox[1], word.bbox[0]))
    lines: list[list[OCRWord]] = []
    for word in ordered:
        if not lines or abs(word.bbox[1] - lines[-1][0].bbox[1]) > max(8, word.bbox[3] - word.bbox[1]):
            lines.append([word])
        else:
            lines[-1].append(word)
    return "\n".join(" ".join(word.text for word in sorted(line, key=lambda item: item.bbox[0])) for line in lines)


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(self, language: str = "eng", config: str = "--oem 3 --psm 3") -> None:
        self.language = language
        self.config = config
        import pytesseract
        self._pytesseract = pytesseract

    def extract(self, image: Image.Image) -> EngineResult:
        data = self._pytesseract.image_to_data(
            image, lang=self.language, config=self.config, output_type=self._pytesseract.Output.DICT
        )
        words: list[OCRWord] = []
        for index, raw_text in enumerate(data["text"]):
            text = raw_text.strip()
            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError):
                confidence = 0.0
            if text and confidence >= 0:
                x, y, width, height = (int(data[key][index]) for key in ("left", "top", "width", "height"))
                words.append(OCRWord(text, confidence, (x, y, x + width, y + height)))
        mean, low_ratio = _metrics(words)
        return EngineResult(_lines_from_words(words), words, mean, low_ratio)


class PaddleOCREngine(OCREngine):
    name = "paddleocr"

    def __init__(self, language: str = "en") -> None:
        import numpy as np
        from paddleocr import PaddleOCR
        self._numpy = np
        self._ocr = PaddleOCR(lang=language, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)

    def extract(self, image: Image.Image) -> EngineResult:
        result = self._ocr.predict(self._numpy.asarray(image))
        words: list[OCRWord] = []
        for page in result:
            data: Any = page.json if hasattr(page, "json") else page
            if callable(data):
                data = data()
            data = data.get("res", data) if isinstance(data, dict) else {}
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            boxes = data.get("rec_boxes", [])
            for text, score, box in zip(texts, scores, boxes):
                values = [int(value) for value in box]
                words.append(OCRWord(str(text).strip(), float(score) * 100, tuple(values)))
        mean, low_ratio = _metrics(words)
        return EngineResult(_lines_from_words(words), words, mean, low_ratio)


class FallbackOCR:
    def __init__(self, engines: list[OCREngine], minimum_confidence: float = 60.0) -> None:
        if not engines:
            raise ValueError("At least one OCR engine is required")
        self.engines = engines
        self.minimum_confidence = minimum_confidence

    def extract(self, image: Image.Image) -> tuple[str, EngineResult]:
        errors: list[str] = []
        best: tuple[str, EngineResult] | None = None
        for engine in self.engines:
            try:
                result = engine.extract(image)
                if result.text.strip() and result.mean_confidence >= self.minimum_confidence:
                    return engine.name, result
                errors.append(f"{engine.name}: low confidence or empty output")
                if result.text.strip() and (best is None or result.mean_confidence > best[1].mean_confidence):
                    best = (engine.name, result)
            except Exception as error:
                errors.append(f"{engine.name}: {error}")
        if best is not None:
            return best
        raise RuntimeError("All OCR engines failed: " + "; ".join(errors))
