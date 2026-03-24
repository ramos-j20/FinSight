"""SEC EDGAR API client for fetching company filings."""
import re
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from backend.core.config import get_settings
from backend.core.exceptions import EDGARFetchError
from backend.core.logging import get_logger

logger = get_logger(__name__)

# SEC EDGAR endpoints
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_BASE_URL = "https://data.sec.gov/submissions"
ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"


@dataclass
class FilingRecord:
    """Represents a single SEC filing record."""

    accession_number: str
    filing_date: str
    period_of_report: str
    primary_document_url: str
    form_type: str


def _is_retryable(exc: BaseException) -> bool:
    """Return True if the exception is retryable (429 or 5xx)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class EDGARClient:
    """Client for the SEC EDGAR API.

    Wraps the EDGAR full-text search and filing APIs, sending the
    required User-Agent header on every request.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._user_agent = settings.EDGAR_USER_AGENT
        self._headers = {
            "User-Agent": self._user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    async def _get(self, url: str) -> httpx.Response:
        """Perform an HTTP GET with retry logic."""
        async with httpx.AsyncClient(headers=self._headers, timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def get_company_cik(self, ticker: str) -> str:
        """Fetch the CIK (Central Index Key) for a given ticker symbol.

        Uses the SEC company_tickers.json endpoint which returns a mapping
        of all tickers to their CIK numbers.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").

        Returns:
            CIK string, zero-padded to 10 digits.

        Raises:
            EDGARFetchError: If the ticker is not found or the API fails.
        """
        logger.info("Fetching CIK for ticker", ticker=ticker)
        try:
            response = await self._get(COMPANY_TICKERS_URL)
            data = response.json()
        except Exception as exc:
            raise EDGARFetchError(
                f"Failed to fetch company tickers from SEC: {exc}"
            ) from exc

        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                logger.info("Found CIK", ticker=ticker, cik=cik)
                return cik

        raise EDGARFetchError(f"Ticker '{ticker}' not found in SEC company tickers")

    async def get_filings(
        self, cik: str, filing_type: str, count: int = 10
    ) -> list[FilingRecord]:
        """Return a list of recent filings of the specified type.

        Args:
            cik: CIK number (zero-padded to 10 digits).
            filing_type: Filing type filter (e.g. "10-K", "10-Q").
            count: Maximum number of filings to return.

        Returns:
            List of FilingRecord objects.

        Raises:
            EDGARFetchError: If the API call fails.
        """
        logger.info(
            "Fetching filings",
            cik=cik,
            filing_type=filing_type,
            count=count,
        )
        cik_padded = cik.zfill(10)
        url = f"{SUBMISSIONS_BASE_URL}/CIK{cik_padded}.json"

        try:
            response = await self._get(url)
            data = response.json()
        except Exception as exc:
            raise EDGARFetchError(
                f"Failed to fetch submissions for CIK {cik}: {exc}"
            ) from exc

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])

        records: list[FilingRecord] = []
        cik_int = str(int(cik_padded))  # remove leading zeros for URL

        for i, form in enumerate(forms):
            if form == filing_type and len(records) < count:
                accession = accession_numbers[i]
                accession_no_dashes = accession.replace("-", "")
                doc_url = (
                    f"{ARCHIVES_BASE_URL}/{cik_int}/"
                    f"{accession_no_dashes}/{primary_docs[i]}"
                )
                records.append(
                    FilingRecord(
                        accession_number=accession,
                        filing_date=filing_dates[i],
                        period_of_report=report_dates[i] if i < len(report_dates) else "",
                        primary_document_url=doc_url,
                        form_type=form,
                    )
                )

        logger.info(
            "Found filings",
            cik=cik,
            filing_type=filing_type,
            count=len(records),
        )
        return records

    async def fetch_filing_text(self, filing_record: FilingRecord) -> str:
        """Download and extract plain text from a filing document.

        Handles both HTML and plain text formats. HTML content is parsed
        with BeautifulSoup to strip tags.

        Args:
            filing_record: A FilingRecord with the URL to fetch.

        Returns:
            Clean plain text of the filing.

        Raises:
            EDGARFetchError: If the download or parsing fails.
        """
        url = filing_record.primary_document_url
        logger.info("Fetching filing text", url=url)

        try:
            response = await self._get(url)
            raw = response.text
        except Exception as exc:
            raise EDGARFetchError(
                f"Failed to download filing from {url}: {exc}"
            ) from exc

        # Detect HTML content and strip tags
        if bool(re.search(r"<\s*(html|body|div|table|p)\b", raw, re.IGNORECASE)):
            soup = BeautifulSoup(raw, "html.parser")
            
            # Preserve tabular structure with spaces
            for td in soup.find_all(["td", "th"]):
                td.append("    ")
            # Ensure block elements start new lines
            for block in soup.find_all(["tr", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "br"]):
                block.append("\n")
                
            text = soup.get_text(separator=" ", strip=True)
            # Normalize excessive whitespace but keep enough for tables
            text = re.sub(r' {3,}', '   ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
        else:
            text = raw

        # Basic whitespace normalization
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        logger.info(
            "Filing text extracted",
            accession=filing_record.accession_number,
            char_count=len(text),
        )
        return text
