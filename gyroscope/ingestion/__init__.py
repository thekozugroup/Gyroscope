"""Phase 1: Ingestion.

Turns filesystem paths and URLs into a list of :class:`Document`.
"""

from __future__ import annotations

from gyroscope.ingestion.base import Loader, LoaderRegistry
from gyroscope.ingestion.docx import DocxLoader
from gyroscope.ingestion.html import HtmlLoader
from gyroscope.ingestion.markdown import MarkdownLoader
from gyroscope.ingestion.pdf import PdfLoader
from gyroscope.ingestion.pipeline import IngestionPipeline
from gyroscope.ingestion.txt import TxtLoader
from gyroscope.ingestion.web import WebLoader

__all__ = [
    "DocxLoader",
    "HtmlLoader",
    "IngestionPipeline",
    "Loader",
    "LoaderRegistry",
    "MarkdownLoader",
    "PdfLoader",
    "TxtLoader",
    "WebLoader",
]
