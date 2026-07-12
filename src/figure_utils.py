from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import BinaryIO, Iterable, Union

import fitz

from src.models import FigureExplanation, FigureRecord
from src.text_utils import clean_text, split_sentences


PdfInput = Union[str, Path, bytes, bytearray, BinaryIO]

CAPTION_START = re.compile(
    r"^\s*(?:fig(?:ure)?s?\.?\s*[a-z]?\d+|图\s*[a-z]?\d+)",
    flags=re.IGNORECASE,
)


def _read_pdf_bytes(file_obj: PdfInput) -> bytes:
    if isinstance(file_obj, (bytes, bytearray)):
        return bytes(file_obj)
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj).read_bytes()

    original_position = None
    try:
        original_position = file_obj.tell()
    except Exception:
        pass

    try:
        file_obj.seek(0)
    except Exception:
        pass

    data = file_obj.read()

    if original_position is not None:
        try:
            file_obj.seek(original_position)
        except Exception:
            pass

    return data


def _caption_candidates(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    candidates: list[tuple[fitz.Rect, str]] = []
    for block in page.get_text("blocks"):
        text = clean_text(str(block[4])).replace("\n", " ")
        if text and CAPTION_START.match(text):
            candidates.append((fitz.Rect(block[:4]), text))
    return candidates


def _nearest_caption(
    image_rect: fitz.Rect,
    captions: Iterable[tuple[fitz.Rect, str]],
) -> str | None:
    ranked: list[tuple[float, str]] = []
    for caption_rect, caption in captions:
        if caption_rect.y0 >= image_rect.y1:
            distance = caption_rect.y0 - image_rect.y1
        elif caption_rect.y1 <= image_rect.y0:
            distance = (image_rect.y0 - caption_rect.y1) + 35.0
        else:
            distance = abs(caption_rect.y0 - image_rect.y1) + 15.0
        ranked.append((distance, caption))

    if not ranked:
        return None

    distance, caption = min(ranked, key=lambda item: item[0])
    return caption if distance <= 220 else None


def _nearby_text(page: fitz.Page, image_rect: fitz.Rect, caption: str | None) -> str:
    snippets: list[str] = []
    search_area = fitz.Rect(
        0,
        max(0, image_rect.y0 - 180),
        page.rect.width,
        min(page.rect.height, image_rect.y1 + 260),
    )
    for block in page.get_text("blocks"):
        block_rect = fitz.Rect(block[:4])
        if not block_rect.intersects(search_area):
            continue
        text = clean_text(str(block[4])).replace("\n", " ")
        if text and text != caption:
            snippets.append(text)

    complete_sentences = split_sentences(" ".join(snippets))
    return " ".join(complete_sentences[:4])[:1600]


def extract_figures_from_pdf(
    file_obj: PdfInput,
    document_name: str,
    document_id: str | None = None,
    min_width: int = 80,
    min_height: int = 80,
    min_area: int = 12_000,
) -> list[FigureRecord]:
    """Extract meaningful raster images and nearby captions from a PDF.

    The input is copied to bytes first, so a file-like upload remains reusable by
    the existing text extraction path. Page numbers in returned records are
    one-based for display and citation.
    """
    pdf_bytes = _read_pdf_bytes(file_obj)
    if not pdf_bytes:
        return []

    stable_document_id = document_id or hashlib.sha256(pdf_bytes).hexdigest()[:16]
    figures: list[FigureRecord] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc):
            captions = _caption_candidates(page)
            page_image_index = 0

            for image_info in page.get_images(full=True):
                xref = int(image_info[0])
                extracted = doc.extract_image(xref)
                width = int(extracted.get("width", 0))
                height = int(extracted.get("height", 0))

                if (
                    width < min_width
                    or height < min_height
                    or width * height < min_area
                ):
                    continue

                rects = page.get_image_rects(xref)
                image_rect = rects[0] if rects else page.rect
                caption = _nearest_caption(image_rect, captions)
                page_image_index += 1

                figures.append(
                    FigureRecord(
                        document_id=stable_document_id,
                        document_name=document_name,
                        page_number=page_index + 1,
                        image_index=page_image_index,
                        width=width,
                        height=height,
                        image_bytes=extracted["image"],
                        extension=str(extracted.get("ext", "png")),
                        caption=caption,
                        nearby_text=_nearby_text(page, image_rect, caption),
                    )
                )

    return figures


def explain_figure(
    figure: FigureRecord,
    language: str = "English",
) -> FigureExplanation:
    """Explain only what caption and nearby page text support.

    This baseline deliberately does not inspect pixels. Exact values, axes,
    colors, and line styles stay unknown unless the caption states them.
    """
    chinese = language == "中文"
    direct: list[str] = []
    inference: list[str] = []

    if figure.caption:
        direct.append(
            f"原文 caption：{figure.caption}"
            if chinese
            else f"Original caption: {figure.caption}"
        )

    nearby = clean_text(figure.nearby_text)
    if nearby:
        direct.append(
            f"同页邻近文字：{nearby[:600]}"
            if chinese
            else f"Nearby text on the same page: {nearby[:600]}"
        )

    evidence_text = " ".join(filter(None, [figure.caption, nearby])).lower()
    if figure.caption:
        if any(term in evidence_text for term in ("schematic", "diagram", "示意图")):
            inference.append(
                "caption 表明该图用于说明结构、流程或实验布局。"
                if chinese
                else "The caption indicates that the figure explains a structure, process, or experimental layout."
            )
        elif any(term in evidence_text for term in ("versus", "comparison", "compare", "dependence", "as a function")):
            inference.append(
                "caption 表明该图用于比较对象或展示变量之间的关系。"
                if chinese
                else "The caption indicates that the figure compares items or shows a relationship between variables."
            )
        elif any(term in evidence_text for term in ("spectrum", "spectra", "micrograph", "mapping", "map", "image")):
            inference.append(
                "caption 表明该图呈现实验图像、映射或谱学结果；具体视觉特征仍需查看原图。"
                if chinese
                else "The caption identifies an image, map, or spectrum; specific visual features still require inspection of the figure."
            )
        else:
            inference.append(
                "该图在论文论证中的作用只能依据 caption 初步判断，需结合正文核实。"
                if chinese
                else "The figure's role in the paper can only be inferred provisionally from its caption and should be checked against the main text."
            )

    if not direct:
        direct.append(
            "未检测到可用于解释该图的 caption 或邻近文字。"
            if chinese
            else "No caption or nearby text was detected for a grounded explanation."
        )

    unknown = (
        (
            "坐标轴含义、精确数值、颜色和线型不能由当前文本证据确认。",
            "图像中未写入 caption 的趋势或对象不能安全推断。",
        )
        if chinese
        else (
            "Axis meanings, exact values, colors, and line styles cannot be confirmed from the current text evidence.",
            "Trends or objects not stated in the caption cannot be inferred safely from this text-only baseline.",
        )
    )

    return FigureExplanation(tuple(direct), tuple(inference), tuple(unknown))
