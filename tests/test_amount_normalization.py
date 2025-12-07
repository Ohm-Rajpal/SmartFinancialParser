import pytest
from src.normalization.amount_normalizer import AmountNormalizer

class TestAmountNormalizer:

    @pytest.fixture
    def normalizer(self):
        return AmountNormalizer()

    @pytest.mark.parametrize("input_str,expected_value,expected_currency", [
        ("$457.00", 457.00, "USD"),
        ("€ 469.10", 469.10, "EUR"),
        ("£459,00", 459.00, "GBP"),
        ("¥1,234.56", 1234.56, "JPY"),
        ("301.73 USD", 301.73, "USD"),
        ("EUR 789.45", 789.45, "EUR"),
        ("-EUR301.73", -301.73, "EUR"),
        ("1+1,177.15 USD", 1177.15, "USD"),
        ("AUD1234.56", 1234.56, "AUD"),
        ("₹ 5,67,890.00", 567890.00, "₹"),
        ("  $  1,234.00 ", 1234.00, "USD"),
        ('"€1.234,56"', 1234.56, "EUR"),
        ("-$1,234.56", -1234.56, "USD"),
        ("-¥1,234", -1234.0, "JPY"),
        ("₩1,000,000", 1000000.0, "₩"),
        ("CHF 123.45", 123.45, "CHF"),
        ("₽ 4.567,89", 4567.89, "₽"),
        ("NZD 9876", 9876.0, "NZD"),
        ("$888888888888888888888", 8.888888888888889e+20, "USD"),
        ("-€999999999999999999999", -1e+18, "EUR"),
        ("++$123.45", 123.45, "USD"),
        ("$1..234,56", 1234.56, "USD"),
        ("€--1234.56", -1234.56, "EUR"),
        ("$12 34 56", 123456.0, "USD"),
        ("USD1,23.4+5", 1234.5, "USD"),
        ("¤1234.56", 1234.56, "¤"),
        ("XYZ 1,2,3,4,5.67", 12345.67, "XYZ"),
    ])
    
    def test_normalize_amounts(self, normalizer, input_str, expected_value, expected_currency):
        result = normalizer.normalize(input_str)
        assert result is not None
        value, currency = result
        assert value == pytest.approx(expected_value, 0.01)
        assert currency == expected_currency

    @pytest.mark.parametrize("input_str", [
        "", "not a number", "$$$", "abc123", None
    ])
    def test_invalid_amounts(self, normalizer, input_str):
        result = normalizer.normalize(input_str)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
