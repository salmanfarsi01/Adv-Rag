from PIL import Image

from ocr_pipeline.engines import EngineResult, FallbackOCR, OCREngine
from ocr_pipeline.models import OCRWord


class StubEngine(OCREngine):
    def __init__(self, name, result=None, error=None):
        self.name = name
        self.result = result
        self.error = error

    def extract(self, image):
        if self.error:
            raise self.error
        return self.result


def result(text, confidence):
    word = OCRWord(text, confidence, (0, 0, 10, 10))
    return EngineResult(text, [word], confidence, 0.0)


def test_falls_back_on_low_confidence():
    ocr = FallbackOCR([StubEngine("tesseract", result("weak", 20)), StubEngine("paddleocr", result("strong", 95))])
    engine, output = ocr.extract(Image.new("RGB", (20, 20)))
    assert engine == "paddleocr"
    assert output.text == "strong"


def test_falls_back_when_primary_raises():
    ocr = FallbackOCR([StubEngine("tesseract", error=RuntimeError("not installed")), StubEngine("paddleocr", result("ok", 90))])
    engine, output = ocr.extract(Image.new("RGB", (20, 20)))
    assert engine == "paddleocr"
    assert output.text == "ok"
