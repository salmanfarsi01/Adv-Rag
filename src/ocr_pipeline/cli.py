from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .engines import FallbackOCR, PaddleOCREngine, TesseractEngine
from .pipeline import PDFOCRPipeline, write_jsonl
from .reporting import write_accuracy_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch OCR English PDFs with automatic fallback")
    parser.add_argument("input", type=Path, help="PDF file or directory containing PDFs")
    parser.add_argument("--output", type=Path, default=Path("data/output/ocr.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("reports/ocr_accuracy.csv"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=60.0)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    paths = sorted(args.input.glob("*.pdf")) if args.input.is_dir() else [args.input]
    if not paths:
        raise SystemExit("No PDF files found")
    engines = []
    try:
        engines.append(PaddleOCREngine())
    except Exception as error:
        logging.warning("PaddleOCR is unavailable; using Tesseract fallback: %s", error)
    try:
        engines.append(TesseractEngine())
    except Exception as error:
        logging.warning("Tesseract is unavailable: %s", error)
    if not engines:
        raise SystemExit("No OCR engine is available")
    pipeline = PDFOCRPipeline(FallbackOCR(engines, args.threshold), args.dpi)
    documents = []
    for path in paths:
        logging.info("Processing %s", path)
        try:
            documents.append(pipeline.process(path))
        except Exception:
            logging.exception("Failed to process %s", path)
    if not documents:
        raise SystemExit("No documents were processed successfully")
    write_jsonl(documents, args.output)
    write_accuracy_report(documents, args.report)
    logging.info("Wrote %s and %s", args.output, args.report)


if __name__ == "__main__":
    main()
