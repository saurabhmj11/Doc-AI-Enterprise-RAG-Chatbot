import pytest
from core.privacy import PrivacyLayer

def test_privacy_masking_basic():
    privacy = PrivacyLayer()
    
    text = "My name is John Doe and my phone number is 555-123-4567. I live in New York."
    masked = privacy.mask_text(text)
    
    assert "John Doe" not in masked
    assert "555-123-4567" not in masked
    assert "New York" not in masked or "<LOCATION>" in masked
    assert "<PERSON>" in masked
    assert "<PHONE>" in masked

def test_privacy_masking_medical():
    privacy = PrivacyLayer()
    
    text = "Patient John Smith (SSN: 123-44-6789) was seen on 2024-02-11."
    masked = privacy.mask_text(text)
    
    assert "John Smith" not in masked
    assert "123-44-6789" not in masked
    assert "2024-02-11" not in masked or "<DATE_TIME>" in masked or "<REDACTED>" in masked

def test_privacy_masking_empty():
    privacy = PrivacyLayer()
    assert privacy.mask_text("") == ""
    assert privacy.mask_text(None) is None
