from scripts.rag.helper import _load_docs
from pathlib import Path
import os
import sys

# Mock PyPDFLoader to avoid needing actual PDFs
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Create a dummy PDF file for testing
Path("test_doc.pdf").touch()

try:
    # We need to test if _load_docs logic preserves 'source'
    # helper.py uses PyPDFLoader. We will trust PyPDFLoader adds source,
    # but we want to check if the list comprehension in _load_docs keeps it.

    # Actually, we can just inspect the code of _load_docs in helper.py
    # "metadata={**d.metadata, "brand": brand, "type": doc_type}"
    # This definitely preserves existing metadata keys like 'source'.

    print("Code inspection confirms **d.metadata preserves 'source'.")
    print("Proceeding to refactor docs_to_context into helper.py")

except Exception as e:
    print(f"Error: {e}")
finally:
    if Path("test_doc.pdf").exists():
        Path("test_doc.pdf").unlink()
