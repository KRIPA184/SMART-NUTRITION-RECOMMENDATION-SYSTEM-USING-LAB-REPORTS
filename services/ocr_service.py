"""
OCR Service – Phase 1
Handles PDF parsing (PyMuPDF) and image OCR (pytesseract).
Returns raw text extracted from uploaded lab reports.
"""

import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from werkzeug.utils import secure_filename


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    Uses PyMuPDF for text-based PDFs; falls back to OCR for scanned pages.
    """
    text_parts: list[str] = []
    doc = fitz.open(file_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")

        if page_text.strip():
            text_parts.append(page_text)
        else:
            # Scanned page – render to image and OCR
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img)
            text_parts.append(ocr_text)

    doc.close()
    return "\n".join(text_parts)


def extract_text_from_image(file_path: str) -> str:
    """Extract text from an image file using Tesseract OCR."""
    img = Image.open(file_path)
    return pytesseract.image_to_string(img)


def process_upload(file_path: str) -> str:
    """
    Determine file type and route to the correct extraction method.
    Returns the raw OCR / parsed text (never sent to LLM directly).
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".png", ".jpg", ".jpeg"):
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
