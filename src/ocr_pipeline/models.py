from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class OCRWord:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass
class OCRPage:
    page_number: int
    width: int
    height: int
    text: str
    words: list[OCRWord]
    engine: str
    mean_confidence: float
    low_confidence_ratio: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["words"] = [asdict(word) for word in self.words]
        return result


@dataclass
class OCRDocument:
    source: str
    pages: list[OCRPage]

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)
