"""
Tests for data validation module.

Tests data quality checks for dates, amounts, merchants, and transactions.
"""

import pytest
from src.validation.data_validator import DataValidator


class TestDataValidator:
    """Test suite for DataValidator"""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance for tests"""
        return DataValidator()
    
    def test_date_validation(self, validator):
        """Test date field validation"""
        # Valid dates (various formats)
        assert validator.validate_date("2024-01-15")[0] is True
        assert validator.validate_date("01/15/2024")[0] is True
        assert validator.validate_date("Jan 15 2024")[0] is True
        
        # Invalid dates
        assert validator.validate_date("")[0] is False
        assert validator.validate_date("   ")[0] is False
        assert validator.validate_date("abc")[0] is False
    
    def test_amount_validation(self, validator):
        """Test amount field validation"""
        # Valid amounts
        assert validator.validate_amount("$123.45")[0] is True
        assert validator.validate_amount("123.45 USD")[0] is True
        assert validator.validate_amount("-50.00")[0] is True
        
        # Invalid amounts
        assert validator.validate_amount("")[0] is False
        assert validator.validate_amount("abc")[0] is False
    
    def test_merchant_validation(self, validator):
        """Test merchant field validation"""
        # Valid merchants
        assert validator.validate_merchant("Starbucks")[0] is True
        assert validator.validate_merchant("McDonald's #123")[0] is True
        
        # Invalid merchants
        assert validator.validate_merchant("")[0] is False
        assert validator.validate_merchant("   ")[0] is False
    
    def test_transaction_validation(self, validator):
        """Test complete transaction validation"""
        # Valid transaction
        txn = {"date": "2024-01-15", "merchant": "Starbucks", "amount": "$12.45"}
        has_critical, issues = validator.validate_transaction(txn)
        assert has_critical is False
        assert len(issues) == 0
        
        # Missing date (critical)
        txn = {"date": "", "merchant": "Target", "amount": "$50.00"}
        has_critical, issues = validator.validate_transaction(txn)
        assert has_critical is True
        
        # Missing merchant (not critical)
        txn = {"date": "2024-01-15", "merchant": "", "amount": "$25.00"}
        has_critical, issues = validator.validate_transaction(txn)
        assert "merchant:missing_merchant" in issues
    
    def test_batch_validation(self, validator):
        """Test batch validation summary"""
        transactions = [
            {"date": "2024-01-15", "merchant": "Starbucks", "amount": "$12.45"},
            {"date": "", "merchant": "Target", "amount": "$50.00"},
            {"date": "2024-01-16", "merchant": "Amazon", "amount": "$25.00"},
        ]
        
        summary = validator.validate_batch(transactions)
        
        assert summary['total'] == 3
        assert summary['valid'] >= 0
        assert summary['critical'] >= 1  # At least one critical (missing date)
    
    def test_completeness_score(self, validator):
        """Test transaction completeness scoring"""
        # Complete
        txn = {"date": "2024-01-15", "merchant": "Starbucks", "amount": "$12.45"}
        assert validator.check_completeness(txn) == 1.0
        
        # Missing one field
        txn = {"date": "2024-01-15", "merchant": "", "amount": "$12.45"}
        assert validator.check_completeness(txn) == pytest.approx(2/3)
        
        # Missing all fields
        txn = {"date": "", "merchant": "", "amount": ""}
        assert validator.check_completeness(txn) == 0.0
    
    def test_data_quality_metrics(self, validator):
        """Test data quality analysis"""
        transactions = [
            {"date": "2024-01-15", "merchant": "Starbucks", "amount": "$12.45"},
            {"date": "", "merchant": "Target", "amount": "$50.00"},
            {"date": "2024-01-16", "merchant": "", "amount": "$25.00"},
        ]
        
        metrics = validator.identify_data_quality_issues(transactions)
        
        assert 'completeness_avg' in metrics
        assert metrics['missing_dates'] == 1
        assert metrics['missing_merchants'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
