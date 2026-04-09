"""
Fetches and downloads papers from arXiv based on a search query.
This tool is taken from https://programmer.ie/post/deepresearch1/
"""
from __future__ import annotations
from typing import Dict, Optional, Type, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

import os
import threading
import time
import requests
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import re

from ..provider_rate_limit import (
    ProviderRateGate,
    ProviderRateLimitTimeout,
    parse_retry_after_seconds,
)

# ---------------------------------------------------------------------------
# Module-level cache for arXiv API responses (metadata only, not PDFs).
# Keyed by (query, max_results). TTL = 30 min.
# ---------------------------------------------------------------------------
_ARXIV_CACHE: Dict[str, tuple[float, str]] = {}
_ARXIV_CACHE_LOCK = threading.Lock()
_ARXIV_CACHE_TTL = 1800  # 30 minutes
_ARXIV_CACHE_MAX = 256


def _arxiv_cache_key(query: str, max_results: int) -> str:
    raw = f"{query.strip().lower()}|{max_results}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _arxiv_cache_get(key: str) -> Optional[str]:
    with _ARXIV_CACHE_LOCK:
        entry = _ARXIV_CACHE.get(key)
        if entry is None:
            return None
        ts, val = entry
        if time.time() - ts > _ARXIV_CACHE_TTL:
            del _ARXIV_CACHE[key]
            return None
        return val


def _arxiv_cache_put(key: str, value: str) -> None:
    with _ARXIV_CACHE_LOCK:
        if len(_ARXIV_CACHE) >= _ARXIV_CACHE_MAX:
            oldest = min(_ARXIV_CACHE, key=lambda k: _ARXIV_CACHE[k][0])
            del _ARXIV_CACHE[oldest]
        _ARXIV_CACHE[key] = (time.time(), value)


SEARCH_QUERY= "agent"  # Replace with desired search term or topic
MAX_RESULTS= 50  # Adjust the number of papers you want to download
OUTPUT_FOLDER= "data"  # Folder to store downloaded papers
BASE_URL= "http://export.arxiv.org/api/query?"


class FetchArxivPapersToolInput(BaseModel):
    search_query: str = Field(description="The search query to use for finding papers on arXiv.")
    max_results: Optional[int] = Field(default=None, description="The maximum number of papers to return.")


class FetchArxivPapersTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "fetch_arxiv_papers"
    description: str = """
    This is a tool will search arxiv based upn the query. I will return a configurable amount of papers ."""
    args_schema: Type[BaseModel] = FetchArxivPapersToolInput
    output_folder: Optional[str] = None

    def __init__(self, working_dir=None, **kwargs: Any):
        """
        Initialize the FetchArxivPapersTool.

        Args:
            working_dir: Optional working directory to save downloaded papers.
                        If not provided, uses the default "data" folder.
        """
        if working_dir:
            output_folder = os.path.join(working_dir, "ideation_agent", "downloaded_papers")
        else:
            output_folder = OUTPUT_FOLDER
        super().__init__(output_folder=output_folder, **kwargs)

    def sanitize_filename(self, title):
        """Sanitizes a string to be used as a filename."""
        # Remove any characters that are not alphanumeric, spaces, hyphens, or underscores
        return re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")

    def get_filename_from_url(self, url):
        # Parse the URL to get the path component
        parsed_url = urlparse(url)
        # Get the base name from the URL's path
        filename = os.path.basename(parsed_url.path)
        return filename

    def compute_file_hash(self, file_path, algorithm="sha256"):
        """Compute the hash of a file using the specified algorithm."""
        hash_func = hashlib.new(algorithm)

        with open(file_path, "rb") as file:
            # Read the file in chunks of 8192 bytes
            while chunk := file.read(8192):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def _request_arxiv(self, url: str, *, stream: bool, action: str, timeout: int = 30):
        gate = ProviderRateGate("arxiv")
        deadline = time.time() + gate.config.max_wait_seconds
        while True:
            remaining = max(deadline - time.time(), 0.0)
            with gate.request(action=action, max_wait_seconds=remaining) as lease:
                try:
                    response = requests.get(url, stream=stream, timeout=timeout)
                except requests.exceptions.Timeout as exc:
                    lease.mark_saturated(f"arXiv request timed out: {exc}")
                    continue
                except Exception as exc:
                    lease.mark_failure(f"arXiv request failed: {exc}")
                    raise

                if response.status_code == 200:
                    lease.mark_success()
                    return response

                if response.status_code in {429, 500, 502, 503, 504}:
                    lease.mark_saturated(
                        f"arXiv returned HTTP {response.status_code}",
                        retry_after_seconds=parse_retry_after_seconds(
                            response.headers.get("Retry-After")
                        ),
                    )
                    continue

                lease.mark_failure(f"arXiv returned HTTP {response.status_code}")
                response.raise_for_status()

    def fetch_arxiv_papers(self, search_query, max_results=5):
        """Fetches metadata of papers from arXiv using the API."""
        url = f"{BASE_URL}search_query=all:{search_query}&start=0&max_results={max_results}"
        response = self._request_arxiv(
            url,
            stream=False,
            timeout=30,
            action=f"arXiv metadata search for '{search_query[:80]}'",
        )
        response.raise_for_status()
        return response.text

    def parse_paper_links(self, response_text):
        """Parses paper links and titles from arXiv API response XML."""
        import xml.etree.ElementTree as ET

        root = ET.fromstring(response_text)
        papers = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            pdf_link = None
            for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                if link.attrib.get("title") == "pdf":
                    pdf_link = link.attrib["href"] + ".pdf"
                    break
            if pdf_link:
                title = self.get_filename_from_url(pdf_link)
                print(title)
                papers.append((title, pdf_link))
        return papers

    def download_paper(self, title, pdf_link, output_folder):
        """Downloads a single paper PDF."""
        # Create a safe filename
        safe_title = self.sanitize_filename(title)
        filename = os.path.join(output_folder, f"{safe_title}.pdf")
        response = self._request_arxiv(
            pdf_link,
            stream=True,
            timeout=60,
            action=f"arXiv PDF download for '{title[:80]}'",
        )
        response.raise_for_status()

        # Write the PDF to the specified folder
        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded: {title}")

    def _run(self, search_query: str, max_results: int = 5):
        # Check cache for previously downloaded results with same query
        ck = _arxiv_cache_key(search_query, max_results)
        cached = _arxiv_cache_get(ck)
        if cached is not None:
            return cached

        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Fetch and parse papers
        print(f"Searching for papers on '{search_query}'...")
        response_text = self.fetch_arxiv_papers(search_query, max_results)
        papers = self.parse_paper_links(response_text)

        # Download each paper (skip if already exists on disk)
        print(f"Found {len(papers)} papers. Starting download...")
        downloaded_papers = []
        for title, pdf_link in papers:
            try:
                safe_title = self.sanitize_filename(title)
                filename = os.path.join(self.output_folder, f"{safe_title}.pdf")
                if os.path.exists(filename):
                    downloaded_papers.append(filename)
                    continue
                self.download_paper(title, pdf_link, self.output_folder)
                downloaded_papers.append(filename)
            except Exception as e:
                if isinstance(e, ProviderRateLimitTimeout):
                    raise
                print(f"Failed to download '{title}': {e}")

        result = f"Download complete! Saved {len(downloaded_papers)} papers to the '{self.output_folder}' directory: {downloaded_papers}"
        _arxiv_cache_put(ck, result)
        return result
