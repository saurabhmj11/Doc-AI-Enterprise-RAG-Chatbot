import os
import pytest
from pathlib import Path

# Force test-specific environment variables before any code is imported
os.environ["USERS_FILE"] = "./users_test.json"
os.environ["ADMIN_USERNAME"] = "admin"
os.environ["ADMIN_PASSWORD"] = "admin"

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_db():
    """Ensure a clean test user database before and after test execution."""
    test_db = Path("./users_test.json")
    if test_db.exists():
        try:
            test_db.unlink()
        except Exception:
            pass
            
    # Seed the admin user now that the file has been cleaned
    from core.auth import _ensure_default_admin
    _ensure_default_admin()
            
    yield
    
    if test_db.exists():
        try:
            test_db.unlink()
        except Exception:
            pass
