"""
Test database error recovery mechanisms.

Tests retry logic, exponential backoff, and error logging for database operations.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.exc import OperationalError, IntegrityError
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import execute_with_retry

# Configure pytest to use anyio for async tests
pytestmark = pytest.mark.anyio


class TestDatabaseRecovery:
    """Test database error recovery mechanisms."""
    
    async def test_execute_with_retry_success_first_attempt(self):
        """Test successful operation on first attempt."""
        mock_operation = AsyncMock(return_value="success")
        
        result = await execute_with_retry(mock_operation, "Test Operation", max_retries=3)
        
        assert result == "success"
        assert mock_operation.call_count == 1
    
    async def test_execute_with_retry_success_after_failures(self):
        """Test successful operation after transient failures."""
        mock_operation = AsyncMock(
            side_effect=[
                OperationalError("Connection timeout", None, None),
                OperationalError("Connection timeout", None, None),
                "success"
            ]
        )
        
        result = await execute_with_retry(mock_operation, "Test Operation", max_retries=3)
        
        assert result == "success"
        assert mock_operation.call_count == 3
    
    async def test_execute_with_retry_all_attempts_fail(self):
        """Test operation fails after all retry attempts."""
        mock_operation = AsyncMock(
            side_effect=OperationalError("Connection failed", None, None)
        )
        
        with pytest.raises(OperationalError):
            await execute_with_retry(mock_operation, "Test Operation", max_retries=3)
        
        assert mock_operation.call_count == 3
    
    async def test_exponential_backoff_timing(self):
        """Test exponential backoff delays are applied correctly."""
        mock_operation = AsyncMock(
            side_effect=[
                OperationalError("Connection timeout", None, None),
                OperationalError("Connection timeout", None, None),
                "success"
            ]
        )
        
        start_time = asyncio.get_event_loop().time()
        result = await execute_with_retry(mock_operation, "Test Operation", max_retries=3)
        end_time = asyncio.get_event_loop().time()
        
        # Expected delays: 1s + 2s = 3s (with some tolerance)
        elapsed = end_time - start_time
        assert elapsed >= 3.0, f"Expected at least 3s delay, got {elapsed}s"
        assert elapsed < 4.0, f"Expected less than 4s delay, got {elapsed}s"
        assert result == "success"
    
    async def test_retry_with_different_exceptions(self):
        """Test retry logic handles different exception types."""
        mock_operation = AsyncMock(
            side_effect=[
                IntegrityError("Constraint violation", None, None),
                "success"
            ]
        )
        
        result = await execute_with_retry(mock_operation, "Test Operation", max_retries=3)
        
        assert result == "success"
        assert mock_operation.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
