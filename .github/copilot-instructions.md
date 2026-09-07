# Project Instructions

- Use Python 3.10 or newer.
- Keep OCR engines behind the adapters in `src/ocr_pipeline/engines.py`.
- Preserve page, word, confidence, and bounding-box source mapping in output contracts.
- Keep PaddleOCR as the primary engine and Tesseract 5.x with pytesseract as the fallback for English PDFs.
- Run `python -m compileall -q src tests` after code changes.
- Run `python -m pytest -q` when development dependencies are installed.
