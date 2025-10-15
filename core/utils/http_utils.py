"""HTTP client utilities with retry logic and backoff.

This module provides a shared HTTP session with automatic retries
and exponential backoff for robust API communication.
"""

from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional

from core.config import HTTP_MAX_RETRIES, HTTP_BACKOFF_FACTOR, HTTP_RETRY_STATUS_CODES


# Global session singleton
_session: Optional[Session] = None


def requests_session() -> Session:
    """Get or create a shared requests session with retry/backoff.
    
    Creates a singleton Session object configured with:
    - Automatic retries for transient failures
    - Exponential backoff between retries
    - Retry on specific HTTP status codes (429, 500, 502, 503, 504)
    
    Returns
    -------
    Session
        Configured requests Session object with retry logic.
        
    Notes
    -----
    - Session is reused across calls for connection pooling
    - Retries only on GET, HEAD, OPTIONS (idempotent methods)
    - Uses exponential backoff: wait = backoff_factor * (2 ** retry_number)
    
    Examples
    --------
    >>> session = requests_session()
    >>> response = session.get('https://api.example.com/data')
    """
    global _session
    
    if _session is None:
        session = Session()
        
        retry = Retry(
            total=HTTP_MAX_RETRIES,
            backoff_factor=HTTP_BACKOFF_FACTOR,
            status_forcelist=HTTP_RETRY_STATUS_CODES,
            allowed_methods=["GET", "HEAD", "OPTIONS"],
        )
        
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        _session = session
    
    return _session


def reset_session() -> None:
    """Reset the global session singleton.
    
    Useful for testing or forcing a fresh session configuration.
    """
    global _session
    if _session is not None:
        _session.close()
        _session = None
