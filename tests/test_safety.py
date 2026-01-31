import pytest
import pandas as pd
from datetime import datetime, timedelta
from ml.safety import CorrectionValidator


@pytest.fixture
def validator():
    return CorrectionValidator()


def test_format_validation_valid(validator):
    """Test that valid formats pass validation"""
    correction = {
        "corrected_fields": {
            "total": 1500.75,
            "invoice_date": "2026-01-15"
        }
    }
    assert validator._format_valid(correction)


def test_format_validation_invalid_total(validator):
    """Test that invalid totals fail validation"""
    correction = {
        "corrected_fields": {
            "total": -100.0,  # Negative amount
            "invoice_date": "2026-01-15"
        }
    }
    assert not validator._format_valid(correction)


def test_format_validation_invalid_date(validator):
    """Test that invalid dates fail validation"""
    correction = {
        "corrected_fields": {
            "total": 1500.75,
            "invoice_date": "invalid-date"
        }
    }
    assert not validator._format_valid(correction)


def test_temporal_quarantine_pass(validator):
    """Test that old enough corrections pass quarantine check"""
    correction = {
        "created_at": datetime.utcnow() - timedelta(hours=73)  # 73 hours ago
    }
    assert validator._temporal_quarantine(correction)


def test_temporal_quarantine_fail(validator):
    """Test that recent corrections fail quarantine check"""
    correction = {
        "created_at": datetime.utcnow() - timedelta(hours=71)  # 71 hours ago
    }
    assert not validator._temporal_quarantine(correction)


def test_confidence_gap_check_pass(validator):
    """Test that corrections to low-confidence predictions pass"""
    correction = {
        "field_confidences": {"total": 0.4},
        "corrected_fields": {"total": 1500.75}
    }
    assert validator._confidence_gap_check(correction)


def test_confidence_gap_check_fail(validator):
    """Test that corrections to high-confidence predictions fail"""
    correction = {
        "field_confidences": {"total": 0.95},  # Very high confidence
        "corrected_fields": {"total": 1500.75}
    }
    assert not validator._confidence_gap_check(correction)


def test_is_training_eligible(validator):
    """Test the full eligibility check"""
    correction = {
        "corrected_fields": {
            "total": 1500.75,
            "invoice_date": "2026-01-15"
        },
        "field_confidences": {"total": 0.4},
        "created_at": datetime.utcnow() - timedelta(hours=73),
        "vendor": "test_vendor"
    }
    
    # Mock historical stats to pass outlier check
    historical_stats = {}
    
    assert validator.is_training_eligible(correction, historical_stats)