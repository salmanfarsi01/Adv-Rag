# English PDF OCR Pipeline

Batch OCR for English PDFs with PaddleOCR as the primary engine and Tesseract 5.x as an automatic local fallback. Each page is rendered at a controlled DPI and emitted as a source-mapped chunk with page number, image dimensions, word text, confidence, and bounding box.

## Setup

1. Create and activate a virtual environment.
2. Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Install Tesseract 5.x separately and ensure `tesseract.exe` is on `PATH`.
4. If PaddleOCR is not installed, install the primary engine and development tools with:

```powershell
python -m pip install -e ".[paddle,dev]"
```

PaddleOCR downloads its English model on first use. CPU execution is supported; GPU configuration is intentionally left to the Paddle installation.

## Run a sample

Place 50-100 representative PDFs in `data/input`, including scans, handwriting, and tables, then run:

```powershell
ocr-pipeline data/input --output data/output/ocr.jsonl --report reports/ocr_accuracy.csv
```

The JSONL output contains one record per page. `words[].bbox` is `[left, top, right, bottom]` in pixels in the rendered page image. This makes each extracted chunk traceable to its source page and region. Text is reconstructed in reading order while retaining word-level geometry for tables and later layout processing.

Useful options:

```powershell
ocr-pipeline data/input --dpi 250 --threshold 65 --log-level INFO
```

PaddleOCR runs first. When it is unavailable or returns empty or below-threshold output, the pipeline calls Tesseract 5.x immediately. If both engines produce low-confidence text, the best non-empty result is retained; if both fail, the file is logged and processing continues with the remaining PDFs.

## Sprint 1 evaluation

The generated `reports/ocr_accuracy.csv` reports mean confidence, low-confidence ratio, word count, character count, engine used, and source page. For a true accuracy score, add a reviewed transcription for each sample page and compare it with the JSONL text using a word or character error-rate script. Confidence is a routing signal, not a substitute for ground truth.

Record failure patterns by filtering the report and reviewing source pages for:

- low scan quality and skew
- handwriting
- dense or merged table cells
- unusual fonts or low contrast
- pages routed to Tesseract fallback

## Development

```powershell
python -m pytest -q
python -m compileall -q src tests
```

The fallback behavior is unit-tested with stub engines, so tests do not require Tesseract or PaddleOCR. Real PDF processing requires the runtime dependencies and external Tesseract installation described above.
