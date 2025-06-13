import chromadb
import pytesseract
from PIL import Image
import logging
from pathlib import Path
from typing import List
from llama_index.core import Document

def ocr_image_to_text(file_path: str) -> List[Document]:
    """Perform OCR on image files using pytesseract"""
    documents = []
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        logging.info(f"OCR SUCCESS: {file_path}\nExtracted text:\n{text[:200]}...")
        doc = Document(
            text=text,
            metadata={
                "file_path": file_path,
                "file_type": "IMAGE",
                "ocr_engine": "Tesseract",
                "ocr_text": text
            }
        )
        documents.append(doc)
    except Exception as e:
        logging.error(f"OCR FAILED: {file_path} - {str(e)}")
    return documents