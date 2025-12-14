"""
CLOUD STORAGE - Backup and restore data to cloud storage

Supports:
- AWS S3
- Google Cloud Storage
- Azure Blob Storage
- Local file backup (for development)

Used for:
1. Database backups (SQLite to cloud)
2. Historical market data caching
3. Knowledge base persistence
4. System state snapshots
"""

import os
import json
import gzip
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, BinaryIO
from abc import ABC, abstractmethod

from loguru import logger


class CloudStorageProvider(ABC):
    """Abstract base class for cloud storage providers."""

    @abstractmethod
    def upload(self, local_path: str, remote_key: str) -> bool:
        """Upload a file to cloud storage."""
        pass

    @abstractmethod
    def download(self, remote_key: str, local_path: str) -> bool:
        """Download a file from cloud storage."""
        pass

    @abstractmethod
    def exists(self, remote_key: str) -> bool:
        """Check if a file exists in cloud storage."""
        pass

    @abstractmethod
    def list_files(self, prefix: str) -> List[str]:
        """List files with a given prefix."""
        pass

    @abstractmethod
    def delete(self, remote_key: str) -> bool:
        """Delete a file from cloud storage."""
        pass


class LocalStorageProvider(CloudStorageProvider):
    """Local file system storage (for development and testing)."""

    def __init__(self, base_path: str = "backups"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorageProvider initialized at {self.base_path}")

    def _resolve_path(self, key: str) -> Path:
        """Resolve a key to a local path."""
        return self.base_path / key

    def upload(self, local_path: str, remote_key: str) -> bool:
        try:
            dest = self._resolve_path(remote_key)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest)
            logger.info(f"Uploaded {local_path} to {remote_key}")
            return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def download(self, remote_key: str, local_path: str) -> bool:
        try:
            src = self._resolve_path(remote_key)
            if not src.exists():
                logger.warning(f"Remote file not found: {remote_key}")
                return False
            shutil.copy2(src, local_path)
            logger.info(f"Downloaded {remote_key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def exists(self, remote_key: str) -> bool:
        return self._resolve_path(remote_key).exists()

    def list_files(self, prefix: str) -> List[str]:
        base = self._resolve_path(prefix)
        if not base.exists():
            return []
        return [str(p.relative_to(self.base_path)) for p in base.rglob("*") if p.is_file()]

    def delete(self, remote_key: str) -> bool:
        try:
            path = self._resolve_path(remote_key)
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False


class S3StorageProvider(CloudStorageProvider):
    """AWS S3 storage provider."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")

        try:
            import boto3
            from botocore.config import Config

            config = Config(
                region_name=region or os.environ.get("AWS_REGION", "ap-northeast-2"),
                retries={"max_attempts": 3, "mode": "adaptive"},
            )

            self.s3 = boto3.client(
                "s3",
                aws_access_key_id=access_key or os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY"),
                config=config,
            )

            logger.info(f"S3StorageProvider initialized for bucket: {bucket}")

        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")

    def _full_key(self, key: str) -> str:
        """Get full S3 key including prefix."""
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def upload(self, local_path: str, remote_key: str) -> bool:
        try:
            full_key = self._full_key(remote_key)
            self.s3.upload_file(local_path, self.bucket, full_key)
            logger.info(f"Uploaded {local_path} to s3://{self.bucket}/{full_key}")
            return True
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False

    def download(self, remote_key: str, local_path: str) -> bool:
        try:
            full_key = self._full_key(remote_key)
            self.s3.download_file(self.bucket, full_key, local_path)
            logger.info(f"Downloaded s3://{self.bucket}/{full_key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return False

    def exists(self, remote_key: str) -> bool:
        try:
            full_key = self._full_key(remote_key)
            self.s3.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except:
            return False

    def list_files(self, prefix: str) -> List[str]:
        try:
            full_prefix = self._full_key(prefix)
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=full_prefix)
            return [obj["Key"] for obj in response.get("Contents", [])]
        except Exception as e:
            logger.error(f"S3 list failed: {e}")
            return []

    def delete(self, remote_key: str) -> bool:
        try:
            full_key = self._full_key(remote_key)
            self.s3.delete_object(Bucket=self.bucket, Key=full_key)
            return True
        except Exception as e:
            logger.error(f"S3 delete failed: {e}")
            return False


class CloudStorage:
    """
    High-level cloud storage manager.

    Provides convenient methods for backing up and restoring
    various types of system data.
    """

    def __init__(self, provider: Optional[CloudStorageProvider] = None):
        """
        Initialize cloud storage.

        Args:
            provider: Storage provider to use. If None, auto-detect from environment.
        """
        if provider:
            self._provider = provider
        else:
            self._provider = self._auto_detect_provider()

    def _auto_detect_provider(self) -> CloudStorageProvider:
        """Auto-detect storage provider from environment variables."""
        # Check for S3
        s3_bucket = os.environ.get("S3_BUCKET") or os.environ.get("AWS_S3_BUCKET")
        if s3_bucket:
            logger.info(f"Using S3 storage: {s3_bucket}")
            return S3StorageProvider(
                bucket=s3_bucket,
                prefix=os.environ.get("S3_PREFIX", "kismutant"),
            )

        # Default to local storage
        logger.info("No cloud storage configured, using local backup")
        return LocalStorageProvider()

    # ===== DATABASE BACKUP =====

    def backup_database(self, db_path: str = "memory/trading_system.db") -> Optional[str]:
        """
        Backup SQLite database to cloud storage.

        Args:
            db_path: Path to SQLite database file

        Returns:
            Remote key if successful, None otherwise
        """
        if not os.path.exists(db_path):
            logger.warning(f"Database file not found: {db_path}")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        remote_key = f"db_backups/trading_system_{timestamp}.db.gz"

        # Compress the database
        compressed_path = f"/tmp/trading_system_{timestamp}.db.gz"
        try:
            with open(db_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            if self._provider.upload(compressed_path, remote_key):
                logger.info(f"Database backed up to {remote_key}")
                return remote_key
            return None

        finally:
            if os.path.exists(compressed_path):
                os.remove(compressed_path)

    def restore_database(
        self,
        remote_key: Optional[str] = None,
        db_path: str = "memory/trading_system.db"
    ) -> bool:
        """
        Restore SQLite database from cloud storage.

        Args:
            remote_key: Specific backup to restore. If None, uses latest.
            db_path: Path to restore database to

        Returns:
            True if successful
        """
        # Find latest backup if not specified
        if not remote_key:
            backups = self._provider.list_files("db_backups/")
            if not backups:
                logger.warning("No database backups found")
                return False
            remote_key = sorted(backups)[-1]  # Latest by name (timestamp)

        compressed_path = f"/tmp/restored_db.gz"
        try:
            if not self._provider.download(remote_key, compressed_path):
                return False

            # Decompress
            with gzip.open(compressed_path, "rb") as f_in:
                with open(db_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            logger.info(f"Database restored from {remote_key}")
            return True

        finally:
            if os.path.exists(compressed_path):
                os.remove(compressed_path)

    # ===== HISTORICAL DATA =====

    def backup_historical_data(self, data: Dict[str, Any], symbol: str) -> Optional[str]:
        """
        Backup historical market data for a symbol.

        Args:
            data: Dictionary containing OHLCV data
            symbol: Stock/ETF symbol

        Returns:
            Remote key if successful
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        remote_key = f"market_data/{symbol}/{symbol}_{timestamp}.json.gz"

        temp_path = f"/tmp/{symbol}_data.json.gz"
        try:
            with gzip.open(temp_path, "wt", encoding="utf-8") as f:
                json.dump(data, f)

            if self._provider.upload(temp_path, remote_key):
                return remote_key
            return None

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def load_historical_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Load historical market data for a symbol.

        Args:
            symbol: Stock/ETF symbol

        Returns:
            Dictionary containing OHLCV data, or None
        """
        # Find latest data file
        prefix = f"market_data/{symbol}/"
        files = self._provider.list_files(prefix)
        if not files:
            return None

        remote_key = sorted(files)[-1]  # Latest by name

        temp_path = f"/tmp/{symbol}_data.json.gz"
        try:
            if not self._provider.download(remote_key, temp_path):
                return None

            with gzip.open(temp_path, "rt", encoding="utf-8") as f:
                return json.load(f)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ===== KNOWLEDGE BASE =====

    def backup_knowledge_base(self, kb_path: str = "memory/knowledge_base.json") -> Optional[str]:
        """Backup knowledge base to cloud storage."""
        if not os.path.exists(kb_path):
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        remote_key = f"knowledge_base/kb_{timestamp}.json.gz"

        temp_path = f"/tmp/kb_backup.json.gz"
        try:
            with open(kb_path, "rb") as f_in:
                with gzip.open(temp_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            if self._provider.upload(temp_path, remote_key):
                return remote_key
            return None

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def restore_knowledge_base(self, kb_path: str = "memory/knowledge_base.json") -> bool:
        """Restore knowledge base from cloud storage."""
        files = self._provider.list_files("knowledge_base/")
        if not files:
            return False

        remote_key = sorted(files)[-1]
        temp_path = f"/tmp/kb_restore.json.gz"

        try:
            if not self._provider.download(remote_key, temp_path):
                return False

            with gzip.open(temp_path, "rb") as f_in:
                with open(kb_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            logger.info(f"Knowledge base restored from {remote_key}")
            return True

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# Singleton instance
_cloud_storage: Optional[CloudStorage] = None


def get_cloud_storage() -> CloudStorage:
    """Get the singleton CloudStorage instance."""
    global _cloud_storage
    if _cloud_storage is None:
        _cloud_storage = CloudStorage()
    return _cloud_storage


# Convenience functions
def backup_all() -> Dict[str, Optional[str]]:
    """Backup all important data to cloud storage."""
    storage = get_cloud_storage()
    return {
        "database": storage.backup_database(),
        "knowledge_base": storage.backup_knowledge_base(),
    }


def restore_all() -> Dict[str, bool]:
    """Restore all data from cloud storage."""
    storage = get_cloud_storage()
    return {
        "database": storage.restore_database(),
        "knowledge_base": storage.restore_knowledge_base(),
    }
