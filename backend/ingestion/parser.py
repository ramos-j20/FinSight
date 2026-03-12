"""SEC filing document parser — cleans raw filing text."""
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedDocument:
    """A cleaned and parsed SEC filing document."""

    ticker: str
    filing_type: str
    period: str
    clean_text: str
    word_count: int
    char_count: int
    parsed_at: datetime


# ---------------------------------------------------------------------------
# Regex patterns for boilerplate removal
# ---------------------------------------------------------------------------
_XBRL_TAG_RE = re.compile(r"<[/]?(?:xbrli?|ix|link|context|us-gaap)[^>]*>", re.IGNORECASE)
_SEC_HEADER_RE = re.compile(
    r"^-----+BEGIN PRIVACY-ENHANCED MESSAGE-----.*?^-----+END PRIVACY-ENHANCED MESSAGE-----",
    re.MULTILINE | re.DOTALL,
)
_PAGE_NUMBER_RE = re.compile(r"^\s*-?\s*\d{1,4}\s*-?\s*$", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# Section-header patterns to preserve (Item 1, Item 1A, PART I, etc.)
_SECTION_HEADER_RE = re.compile(
    r"^(ITEM\s+\d+[A-Z]?\.?.*|PART\s+[IVX]+.*|[A-Z][A-Z\s]{4,})$",
    re.MULTILINE,
)


def parse_filing_to_text(
    raw_text: str,
    filing_type: str,
    ticker: str = "",
    period: str = "",
) -> ParsedDocument:
    """Clean raw filing text and return a ParsedDocument.

    Cleaning pipeline:
        1. Remove XBRL tags and SEC privacy headers/footers.
        2. Remove remaining HTML-like tags.
        3. Remove standalone page-number lines.
        4. Remove non-ASCII characters.
        5. Collapse multiple newlines / spaces.
        6. Preserve section headers (lines in ALL CAPS or "Item …" patterns).

    Args:
        raw_text: The raw filing text (may contain HTML/XBRL).
        filing_type: e.g. "10-K" or "10-Q".
        ticker: Stock ticker (stored in output metadata).
        period: Reporting period string (stored in output metadata).

    Returns:
        A ParsedDocument with cleaned text and statistics.
    """
    logger.info(
        "Parsing filing",
        ticker=ticker,
        filing_type=filing_type,
        raw_char_count=len(raw_text),
    )

    text = raw_text

    # 1. Remove XBRL tags and SEC privacy-enhanced-message blocks
    text = _XBRL_TAG_RE.sub("", text)
    text = _SEC_HEADER_RE.sub("", text)

    # 2. Remove remaining HTML tags
    text = _HTML_TAG_RE.sub("", text)

    # 3. Remove standalone page-number lines
    text = _PAGE_NUMBER_RE.sub("", text)

    # 4. Remove non-ASCII characters
    text = _NON_ASCII_RE.sub("", text)

    # 5. Normalize whitespace
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = text.strip()

    word_count = len(text.split())
    char_count = len(text)

    if char_count < 1000:
        logger.warning(
            "Parsed text is suspiciously short — possible parse failure",
            ticker=ticker,
            filing_type=filing_type,
            char_count=char_count,
        )

    parsed = ParsedDocument(
        ticker=ticker,
        filing_type=filing_type,
        period=period,
        clean_text=text,
        word_count=word_count,
        char_count=char_count,
        parsed_at=datetime.now(timezone.utc),
    )

    logger.info(
        "Filing parsed",
        ticker=ticker,
        filing_type=filing_type,
        word_count=word_count,
        char_count=char_count,
    )
    return parsed
