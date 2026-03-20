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
        """Embed a list of texts using the OpenAI API.

        Texts are processed in batches of 100 to avoid hitting API size limits
        and to ensure regular progress.

        Args:
            texts: The list of texts to embed.

        Returns:
            A list of embedding vectors (one per input text).

        Raises:
            EmbeddingError: If the API call fails.
        """
        batch_size = 100
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            truncated_batch: list[str] = []
            
            for idx, text in enumerate(batch_texts):
                words = text.split()
                if len(words) > _MAX_WORDS:
                    logger.warning(
                        "Text exceeds 2048 token limit, truncating",
                        chunk_index=i + idx,
                        original_words=len(words),
                        truncated_words=_MAX_WORDS,
                    )
                    truncated_batch.append(" ".join(words[:_MAX_WORDS]))
                else:
                    truncated_batch.append(text)

            try:
                response = self._client.embeddings.create(
                    input=truncated_batch,
                    model=MODEL,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.info("Batch embedding complete", batch_start=i, count=len(batch_embeddings))
            except openai.RateLimitError:
                raise  # let tenacity handle retries
            except Exception as exc:
                raise EmbeddingError(f"Failed to embed batch starting at {i}: {exc}") from exc

        return all_embeddings
