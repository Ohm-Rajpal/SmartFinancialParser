"""
Tests for the DateNormalizer module.
"""

import pytest
from src.normalization.date_normalizer import DateNormalizer


class TestDateNormalizer:
    """Test suite for DateNormalizer"""

    @pytest.fixture
    def normalizer(self):
        """Create a DateNormalizer instance"""
        return DateNormalizer()

    @pytest.mark.parametrize("input_date,expected", [
        ("Jan 1st 23", "2023-01-01"),
        ("2nd of Feb, 24", "2024-02-02"),
        ("March 3rd 2025", "2025-03-03"),
        ("04-Jul-21", "2021-07-04"),
        ("5/6/22", "2022-05-06"),
        ("  June   24 , 2025 ", "2025-06-24"),
        ("Nov 4th 2025", "2025-11-04"),
        ("Mar-19-25", "2025-03-19"),
        ("2023-01-01", "2023-01-01"),
        ("12/31/2025", "2025-12-31"),
        ("20250120", "2025-01-20"),
    ])

    def test_normalize_dates(self, normalizer, input_date, expected):
        """Test normalization of various date formats including messy inputs"""
        assert normalizer.normalize_date(input_date) == expected

    @pytest.mark.parametrize("input_date", [
        "", None, "not a date", "32/01/2025", "Feb 30 2025"
    ])
    def test_invalid_dates(self, normalizer, input_date):
        """Test that invalid dates return None"""
        assert normalizer.normalize_date(input_date) is None

    def test_disabled_normalization(self):
        """Test that normalization returns None if disabled"""
        normalizer = DateNormalizer(is_valid_date_format=False)
        assert normalizer.normalize_date("2025-12-06") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
