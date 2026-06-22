"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Literal
from datetime import datetime


# ============ Document Upload ============

class DocumentUploadResponse(BaseModel):
    """Response after document upload and processing."""
    document_id: str
    filename: str
    file_type: str
    chunks_created: int
    status: str = "processed"
    message: str = "Document processed successfully"


# ============ Question Answering ============

class AskRequest(BaseModel):
    """Request to ask a question about one or more documents."""
    document_id: Optional[str] = None
    document_ids: Optional[list[str]] = None
    question: str = Field(..., min_length=3, max_length=1000)


class SourceChunk(BaseModel):
    """A source chunk that supports an answer."""
    text: str
    filename: Optional[str] = None
    page: Optional[int] = None
    chunk_id: str
    similarity_score: float
    hybrid_score: Optional[float] = None      # Set when hybrid search is active
    signal_source: Optional[str] = None       # "dense" | "bm25" | "hybrid"


class AskResponse(BaseModel):
    """Response to a question with answer, sources, and confidence."""
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: str  # "high", "medium", "low"
    sources: list[SourceChunk]
    guardrail_status: str  # "passed", "low_confidence", "not_found"
    guardrail_message: Optional[str] = None


# ============ Structured Extraction ============

class ExtractRequest(BaseModel):
    """Request to extract structured data from a document."""
    document_id: str


class ShipmentData(BaseModel):
    """Structured shipment data extracted from document."""
    shipment_id: Optional[str] = None
    shipper: Optional[str] = None
    consignee: Optional[str] = None
    pickup_datetime: Optional[str] = None
    delivery_datetime: Optional[str] = None
    equipment_type: Optional[str] = None
    mode: Optional[str] = None
    rate: Optional[float] = None
    currency: Optional[str] = None
    weight: Optional[str] = None
    carrier_name: Optional[str] = None


class ExtractResponse(BaseModel):
    """Response with extracted structured data."""
    document_id: str
    extraction: ShipmentData
    extraction_confidence: float
    fields_found: int
    total_fields: int = 11


# ============ Error Responses ============

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None


# ============ Configuration ============

class ConfigResponse(BaseModel):
    """Current configuration settings."""
    llm_mode: str
    gemini_api_key_configured: bool
    ollama_base_url: str
    ollama_model: str


class ConfigUpdate(BaseModel):
    """Configuration update model."""
    llm_mode: str = Field(..., pattern="^(online|offline)$")
    gemini_api_key: Optional[str] = None
    ollama_base_url: Optional[str] = None
    ollama_model: Optional[str] = None


# ============ Debug & Monitoring ============

class ModelInfo(BaseModel):
    """Information about a single LLM model."""
    name: str
    methods: list[str] = []


class ModelsResponse(BaseModel):
    """Response containing a list of available models."""
    models: list[ModelInfo]


# ============ Authentication ============

class Token(BaseModel):
    """JWT Token response."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None


# ============ User Profile ============

class UserProfileResponse(BaseModel):
    """Public user profile (no password hash)."""
    username: str
    full_name: str
    email: str
    department: str = ""   # e.g. "AI & ML", "Platform Engineering"
    designation: str = ""  # e.g. "LLM Engineer", "RAG Engineer", "AI/ML Engineer"
    role: str              # "admin" | "editor" | "viewer"
    disabled: bool = False


class UserProfileUpdate(BaseModel):
    """Fields a user can update on their own profile."""
    full_name: Optional[str] = None
    email: Optional[str] = None
    department: Optional[str] = None
    designation: Optional[str] = None


class UserCreateRequest(BaseModel):
    """Admin-only request to create a new user."""
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(..., min_length=8)
    full_name: str
    email: str
    department: str = Field(..., description="Team / org unit, e.g. 'AI & ML'")
    designation: str = Field(..., description="Job title, e.g. 'LLM Engineer'")
    role: Literal["admin", "editor", "viewer"] = "viewer"
