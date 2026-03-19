"""OpenAI embedding client with retry logic."""
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.core.config import get_settings
from backend.core.exceptions import EmbeddingError
from backend.core.logging import get_logger

logger = get_logger(__name__)

# OpenAI text-embedding-3-small produces 1536-dimensional vectors.
MODEL = "text-embedding-3-small"
DIMENSIONS = 1536

# Approximate token limit per text (OpenAI enforces 8191 tokens for this model,
# but the task spec says 2048).  We use a rough 1 word ≈ 1.3 tokens heuristic
# to avoid adding a tokenizer dependency.
MAX_TOKENS = 2048
_MAX_WORDS = int(MAX_TOKENS / 1.3)  # ~1575 words


class OpenAIEmbedder:
    """Generate embeddings via the OpenAI API.

    Uses ``text-embedding-3-small`` (1536 dimensions) with automatic retry
    on rate-limit errors.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=60),
        retry=retry_if_exception_type(openai.RateLimitError),
        reraise=True,
    )
    def embed_text(self, text: str) -> list[float]:
        """Return the embedding vector for a single text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingError: If the API call fails.
        """
        try:
            response = self._client.embeddings.create(
                input=[text],
                model=MODEL,
            )
            return response.data[0].embedding
        except openai.RateLimitError:
            raise  # let tenacity handle retries
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed text: {exc}") from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=60),
        retry=retry_if_exception_type(openai.RateLimitError),
        reraise=True,
    )
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in a single API call.

        Texts exceeding the 2048-token limit are truncated and a warning
        is logged.

        Args:
            texts: The list of texts to embed.

        Returns:
            A list of embedding vectors (one per input text).

        Raises:
            EmbeddingError: If the API call fails.
        """
        truncated_texts: list[str] = []
        for idx, text in enumerate(texts):
            words = text.split()
            if len(words) > _MAX_WORDS:
                logger.warning(
                    "Text exceeds 2048 token limit, truncating",
                    chunk_index=idx,
                    original_words=len(words),
                    truncated_words=_MAX_WORDS,
                )
                truncated_texts.append(" ".join(words[:_MAX_WORDS]))
            else:
                truncated_texts.append(text)

        try:
            response = self._client.embeddings.create(
                input=truncated_texts,
                model=MODEL,
            )
            # OpenAI returns embeddings sorted by index
            return [item.embedding for item in response.data]
        except openai.RateLimitError:
            raise  # let tenacity handle retries
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed batch: {exc}") from exc
