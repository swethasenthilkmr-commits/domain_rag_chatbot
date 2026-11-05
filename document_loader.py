"""
document_loader.py
Module 1, 2 and 3: validate uploaded PDFs, extract text page by page,
and split the text into overlapping chunks.

Every chunk keeps its metadata: the document name and the page number,
so the chatbot can show where an answer came from.
"""

import io
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_FILE_SIZE_MB = 10
MAX_FILES = 5

# Suggested beginner settings from the project guidelines
CHUNK_SIZE = 800       # characters per chunk (guideline: 700-1000)
CHUNK_OVERLAP = 120    # characters shared by neighbouring chunks (guideline: 100-150)


def validate_files(files):
    """Check the number, type and size of the uploaded files.

    `files` is a list of (file_name, file_bytes). Returns a list of error messages.
    """
    errors = []
    if len(files) > MAX_FILES:
        errors.append(f"Please upload at most {MAX_FILES} PDF files at a time.")
    for name, data in files:
        if not name.lower().endswith(".pdf"):
            errors.append(f"{name}: only PDF files are allowed.")
        elif len(data) == 0:
            errors.append(f"{name}: the file is empty.")
        elif len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
            errors.append(f"{name}: file is larger than {MAX_FILE_SIZE_MB} MB.")
    return errors


def extract_pages(file_name, file_bytes):
    """Read every page of a PDF and return one Document per non-empty page.

    Metadata: {"source": file name, "page": page number starting at 1}.
    Returns (pages, skipped_page_count).
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    pages, skipped = [], 0
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:                 # empty or scanned page - skip safely
            skipped += 1
            continue
        pages.append(Document(page_content=text, metadata={"source": file_name, "page": number}))
    return pages, skipped


def split_into_chunks(pages):
    """Split page texts into smaller overlapping chunks (metadata is kept)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def load_and_chunk(files):
    """Full document step: extract text from all PDFs and split it into chunks.

    `files` is a list of (file_name, file_bytes).
    Returns (chunks, report) where report describes what was processed.
    """
    all_pages, report = [], []
    for name, data in files:
        try:
            pages, skipped = extract_pages(name, data)
        except Exception:
            report.append({"file": name, "pages": 0, "skipped": 0,
                           "status": "Could not be read (damaged or password-protected)"})
            continue
        status = "OK" if pages else "No text found (scanned PDF? OCR is not supported)"
        report.append({"file": name, "pages": len(pages), "skipped": skipped, "status": status})
        all_pages.extend(pages)

    chunks = split_into_chunks(all_pages) if all_pages else []
    return chunks, report
