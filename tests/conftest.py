import pytest
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope='session')
def base_url():
    """Provide base URL for tests"""
    return 'http://localhost:8000'  # Adjust as needed

@pytest.fixture(scope='function')
def clean_up():
    """Fixture for test cleanup"""
    yield
    # Add cleanup code here if needed