"""S3 client for storing and retrieving filing documents."""
import boto3
from botocore.exceptions import ClientError

from backend.core.config import get_settings
from backend.core.exceptions import S3UploadError
from backend.core.logging import get_logger

logger = get_logger(__name__)


class S3Client:
    """Wrapper around boto3 for S3 operations on filing documents.

    S3 key patterns:
        - Raw:       raw/{ticker}/{filing_type}/{period}/filing.txt
        - Processed: processed/{ticker}/{filing_type}/{period}/chunks.json
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._bucket = settings.AWS_S3_BUCKET_NAME
        self._client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )

    def upload_text(self, content: str, s3_key: str) -> str:
        """Upload text content to S3.

        Args:
            content: The text content to upload.
            s3_key: The S3 key to store the content under.

        Returns:
            Full S3 URI (s3://bucket/key).

        Raises:
            S3UploadError: If the upload fails.
        """
        logger.info("Uploading to S3", s3_key=s3_key, bucket=self._bucket)
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=s3_key,
                Body=content.encode("utf-8"),
                ContentType="text/plain",
            )
        except ClientError as exc:
            raise S3UploadError(
                f"Failed to upload to s3://{self._bucket}/{s3_key}: {exc}"
            ) from exc

        uri = f"s3://{self._bucket}/{s3_key}"
        logger.info("Upload complete", s3_uri=uri)
        return uri

    def download_text(self, s3_key: str) -> str:
        """Download and return text content from S3.

        Args:
            s3_key: The S3 key to download from.

        Returns:
            The text content stored at the key.

        Raises:
            S3UploadError: If the download fails.
        """
        logger.info("Downloading from S3", s3_key=s3_key, bucket=self._bucket)
        try:
            response = self._client.get_object(
                Bucket=self._bucket,
                Key=s3_key,
            )
            body = response["Body"].read().decode("utf-8")
        except ClientError as exc:
            raise S3UploadError(
                f"Failed to download s3://{self._bucket}/{s3_key}: {exc}"
            ) from exc

        logger.info("Download complete", s3_key=s3_key, char_count=len(body))
        return body

    def key_exists(self, s3_key: str) -> bool:
        """Check whether an S3 key exists without downloading the content.

        Uses head_object for an efficient existence check.

        Args:
            s3_key: The S3 key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        try:
            self._client.head_object(Bucket=self._bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    @staticmethod
    def raw_key(ticker: str, filing_type: str, period: str) -> str:
        """Build the S3 key for a raw filing."""
        return f"raw/{ticker}/{filing_type}/{period}/filing.txt"

    @staticmethod
    def processed_key(ticker: str, filing_type: str, period: str) -> str:
        """Build the S3 key for processed chunks."""
        return f"processed/{ticker}/{filing_type}/{period}/chunks.json"
