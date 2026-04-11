"""
Connection Pool for SQLite with automatic retry on SQLITE_BUSY.

Provides thread-safe connection pooling to handle concurrent writes
to SQLite database (20+ workers) without blocking on SQLITE_BUSY errors.
"""

import sqlite3
import time
import logging
import threading
from typing import Optional
from queue import Queue, Empty
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class ConnectionPool:
    """
    Thread-safe SQLite connection pool with automatic retry.

    Manages a pool of database connections and automatically retries
    operations that fail with SQLITE_BUSY errors using exponential backoff.
    """

    def __init__(
        self,
        db_path: str,
        pool_size: int = 5,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize connection pool.

        Args:
            db_path: Path to SQLite database file
            pool_size: Maximum number of connections in pool
            timeout: Timeout in seconds for acquiring connection
            max_retries: Maximum number of retry attempts on SQLITE_BUSY
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout
        self.max_retries = max_retries

        self._pool: Queue = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._created_count = 0

        # Pre-create initial connections
        for _ in range(pool_size):
            self._pool.put(self._create_connection())

    def _create_connection(self) -> sqlite3.Connection:
        """
        Create a new database connection with optimal settings.

        Returns:
            SQLite connection object
        """
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False  # Allow connection sharing across threads
        )
        conn.row_factory = sqlite3.Row

        # Try to enable WAL mode for better concurrency, fall back to DELETE mode if it fails
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not enable WAL mode (disk I/O error), using DELETE mode: {e}")
            # WAL mode failed, use default DELETE mode
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=FULL")

        with self._lock:
            self._created_count += 1
            logger.debug(f"Created connection {self._created_count} for {self.db_path}")

        return conn

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a connection from the pool.

        Returns:
            SQLite connection from pool

        Raises:
            Empty: If no connection available within timeout
        """
        try:
            conn = self._pool.get(timeout=self.timeout)
            return conn
        except Empty:
            raise TimeoutError(
                f"Could not acquire connection from pool within {self.timeout}s"
            )

    def _return_connection(self, conn: sqlite3.Connection):
        """
        Return a connection to the pool.

        Args:
            conn: Connection to return to pool
        """
        try:
            self._pool.put(conn, block=False)
        except Exception as e:
            logger.error(f"Failed to return connection to pool: {e}")
            # Connection is lost, create a new one
            try:
                new_conn = self._create_connection()
                self._pool.put(new_conn, block=False)
            except Exception as e2:
                logger.error(f"Failed to create replacement connection: {e2}")

    @contextmanager
    def connection(self):
        """
        Context manager for obtaining a pooled connection with automatic retry.

        Automatically retries operations that fail with SQLITE_BUSY using
        exponential backoff (max 3 attempts per spec).

        Yields:
            SQLite connection from pool

        Example:
            with pool.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO ...")
                conn.commit()
        """
        conn = None
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                if conn is None:
                    conn = self._get_connection()

                yield conn
                return  # Success - exit context manager

            except sqlite3.OperationalError as e:
                last_error = e
                if "database is locked" in str(e).lower() or "busy" in str(e).lower():
                    attempt += 1
                    if attempt < self.max_retries:
                        # Exponential backoff: 0.1s, 0.2s, 0.4s
                        backoff = 0.1 * (2 ** (attempt - 1))
                        logger.warning(
                            f"SQLITE_BUSY error, retrying in {backoff}s "
                            f"(attempt {attempt}/{self.max_retries})"
                        )
                        time.sleep(backoff)
                        continue
                    else:
                        logger.error(
                            f"Failed after {self.max_retries} retries: {e}"
                        )
                        raise
                else:
                    # Not a busy error - raise immediately
                    raise

            except Exception as e:
                last_error = e
                raise

            finally:
                if conn is not None:
                    self._return_connection(conn)
                    conn = None

        # Max retries exceeded
        if last_error:
            raise last_error

    def close_all(self):
        """
        Close all connections in the pool.

        Should be called when shutting down the application.
        """
        closed_count = 0
        while not self._pool.empty():
            try:
                conn = self._pool.get(block=False)
                conn.close()
                closed_count += 1
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

        logger.info(f"Closed {closed_count} connections from pool")

    def __enter__(self):
        """Support using ConnectionPool as context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up connections when exiting context."""
        self.close_all()
