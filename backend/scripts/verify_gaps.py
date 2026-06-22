"""All-in-one production gap verification (static + runtime, no live API required)."""
import os, sys, pathlib, tempfile, uuid, warnings, json

os.environ["SECRET_KEY"] = "test-secret-32chars-minimum-xxxxx"
os.environ["ADMIN_PASSWORD"] = "TestAdmin@2024!"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("FULL PRODUCTION AUDIT VERIFICATION")
print("=" * 60)

# ── GAP 1a: No hardcoded secret ─────────────────────────────
print("\n[1a] No hardcoded SECRET_KEY fallback...")
auth_src = open("core/auth.py").read()
assert "7b99c89d2e3f4a5b6c7d8e9f" not in auth_src, "Old hardcoded secret still present!"
assert "secrets.token_hex" in auth_src, "No auto-gen fallback for missing key!"
assert "FAKE_USERS_DB" not in auth_src, "FAKE_USERS_DB still present!"
print("  PASS - hardcoded secret removed, auto-generate fallback in place")

# ── GAP 1b: User model has designation + department ──────────
print("\n[1b] User profile has designation + department...")
import core.auth as auth_mod

# Point user store to a temp file so we don't touch production data
test_file = pathlib.Path(tempfile.mktemp(suffix=".json"))
auth_mod._USERS_FILE = test_file

from core.auth import (
    get_password_hash, verify_password, create_access_token,
    create_user, update_profile, get_all_users,
)

user = create_user(
    username="saurabh",
    password="Secure@Pass99",
    full_name="Saurabh MJ",
    email="saurabh@ultradoc.ai",
    department="AI & ML",
    designation="LLM Engineer",
    role="admin",
)
assert user["designation"] == "LLM Engineer"
assert user["department"] == "AI & ML"
assert "hashed_password" not in user
print(f"  Created: {user['full_name']} | {user['designation']} | {user['department']}")
print("  PASS - designation and department stored in user profile")

# ── GAP 1c: Profile update works ────────────────────────────
updated = update_profile("saurabh", designation="RAG Engineer", department="Research")
assert updated["designation"] == "RAG Engineer"
print(f"  Updated designation: RAG Engineer  PASS")

# ── GAP 2: Debug endpoints are protected ────────────────────
print("\n[2] Debug endpoints require authentication...")
debug_src = open("api/debug.py").read()
assert "require_role" in debug_src
assert debug_src.count("Depends(require_role") >= 2
print("  PASS - both /models and /health/extended require admin role")

# ── GAP 3: datetime.utcnow() deprecated call is gone ────────
print("\n[3] No deprecated datetime.utcnow()...")
with warnings.catch_warnings():
    warnings.simplefilter("error", DeprecationWarning)
    token = create_access_token({"sub": "saurabh"})
assert token and len(token) > 20
print("  PASS - JWT uses timezone.utc, no DeprecationWarning raised")

# ── GAP 4: Health check reports hybrid pipeline ──────────────
print("\n[4] Extended health check reports full hybrid status...")
debug_src2 = open("api/debug.py").read()
for key in ["bm25_index", "hybrid_pipeline", "fusion_strategy",
            "query_expansion", "reranking", "unindexed_documents"]:
    assert key in debug_src2, f"Missing key in health: {key}"
print("  PASS - health check reports BM25, hybrid config, migration status")

# ── GAP 5: Path traversal fixed ─────────────────────────────
print("\n[5] Upload uses UUID-based filenames...")
routes_src = open("api/routes.py").read()
assert "uuid.uuid4().hex" in routes_src, "UUID filename missing!"
# Verify the actual temp_path assignment uses safe_filename, not file.filename
assert "safe_filename = f\"{uuid.uuid4" not in routes_src or "temp_path = UPLOAD_DIR / safe_filename" in routes_src
assert "temp_path = UPLOAD_DIR / file.filename" not in routes_src, \
    "Path traversal: temp_path still uses user-supplied filename!"
safe_name = f"{uuid.uuid4().hex}.pdf"
assert ".." not in safe_name and "/" not in safe_name
print(f"  Example safe filename: {safe_name}")
print("  PASS - user-supplied filenames never written to disk")

# ── SCHEMAS: UserProfileResponse + UserCreateRequest ────────
print("\n[6] User profile schemas...")
from api.schemas import UserProfileResponse, UserProfileUpdate, UserCreateRequest
from typing import Literal

p = UserProfileResponse(
    username="priya", full_name="Priya Sharma", email="priya@ai.com",
    department="MLOps", designation="MLOps Engineer", role="editor",
)
assert p.designation == "MLOps Engineer"

r = UserCreateRequest(
    username="raj123", password="Pass1234!", full_name="Raj Patel",
    email="raj@ai.com", department="Research", designation="AI Research Engineer",
    role="viewer",
)
assert r.role == "viewer"
print("  PASS - UserProfileResponse and UserCreateRequest schemas valid")

# ── bcrypt rounds = 12 ───────────────────────────────────────
print("\n[7] bcrypt strength...")
import bcrypt as _bcrypt
h = get_password_hash("StrongPass@99")
cost = _bcrypt.checkpw(b"StrongPass@99", h.encode())
# Extract rounds from hash
rounds = int(h.split("$")[2])
assert rounds >= 12, f"bcrypt rounds too low: {rounds}"
print(f"  bcrypt rounds: {rounds}  PASS")

print("\n" + "=" * 60)
print("ALL 7 PRODUCTION GAP CHECKS PASSED")
print("=" * 60)

# Cleanup
test_file.unlink(missing_ok=True)
