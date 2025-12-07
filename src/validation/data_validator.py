"""
Data validation module for checking transaction data quality.

Validates data for:
- Missing or empty fields
- Invalid date formats
- Invalid amount values
- Data type correctness

Flags issues without stopping processing
"""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates transaction data quality"""
    
    def __init__(self):
        """Initialize data validator with config rules"""
        self.min_year = Config.MIN_TRANSACTION_YEAR
        self.max_year = Config.MAX_TRANSACTION_YEAR
        self.min_amount = Config.MIN_TRANSACTION_AMOUNT
        self.max_amount = Config.MAX_TRANSACTION_AMOUNT
    
    def validate_date(self, date_str: str) -> Tuple[bool, Optional[str]]:
        """
        Validate date field.
        
        Checks:
        - Not empty
        - Contains date-like content (has numbers)
        
        Note: Actual parsing happens in date_normalizer.
        This just checks if it looks like a date.
        
        Args:
            date_str: Date string to validate
            
        Returns:
            Tuple of (is_valid, issue_description)
        """
        if not date_str or not date_str.strip():
            return (False, "missing_date")
        
        # Check if contains at least some numbers
        if not any(char.isdigit() for char in date_str):
            return (False, "no_numeric_content")
        
        return (True, None)
    
    def validate_amount(self, amount_str: str) -> Tuple[bool, Optional[str]]:
        """
        Validate amount field.
        
        Checks:
        - Not empty
        - Contains numeric content
        - Reasonable range (after basic parsing)
        
        Note: Actual parsing happens in amount_normalizer.
        This just checks if it looks like an amount.
        
        Args:
            amount_str: Amount string to validate
            
        Returns:
            Tuple of (is_valid, issue_description)
        """
        if not amount_str or not amount_str.strip():
            return (False, "missing_amount")
        
        # Check if contains numeric content
        has_digit = any(char.isdigit() for char in amount_str)
        if not has_digit:
            return (False, "no_numeric_content")
        
        # Try basic extraction to check range
        # Extract all digits and decimal point
        numeric_chars = ''.join(c for c in amount_str if c.isdigit() or c == '.')
        if numeric_chars:
            try:
                # Very basic parse - just check magnitude
                value = float(numeric_chars)
                
                # Check reasonable range
                # min_amount is negative and max_amount is positive
                if value < self.min_amount:
                    return (False, f"amount_too_small_{value}")
                    
                if value > self.max_amount:
                    return (False, f"amount_too_large_{value}")

            except ValueError:
                # If can't parse at all, flag it
                return (False, "unparseable_amount")
        
        return (True, None)
    
    def validate_merchant(self, merchant_str: str) -> Tuple[bool, Optional[str]]:
        """
        Validate merchant field.
        
        Checks:
        - Not empty (after stripping whitespace)
        - Has minimum length
        - Not just whitespace
        
        Args:
            merchant_str: Merchant name to validate
            
        Returns:
            Tuple of (is_valid, issue_description)
        """
        # Handle None, NaN, or non-string types (from pandas)
        if merchant_str is None or (isinstance(merchant_str, float) and pd.isna(merchant_str)):
            return (False, "missing_merchant")
        
        # Convert to string if not already
        if not isinstance(merchant_str, str):
            merchant_str = str(merchant_str)
        
        if not merchant_str:
            return (False, "missing_merchant")
        
        cleaned = merchant_str.strip()
        
        if not cleaned:
            return (False, "empty_merchant")
        
        if len(cleaned) < Config.MIN_MERCHANT_NAME_LENGTH:
            return (False, "merchant_too_short")

        if len(cleaned) > Config.MAX_MERCHANT_NAME_LENGTH:
            return (False, "merchant_too_long")
    
        return (True, None)
    
    def validate_transaction(self, transaction: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate a complete transaction row.
        
        Checks all required fields and returns list of issues.
        Transaction is considered valid even with issues (for graceful degradation).
        
        Args:
            transaction: Dict with keys 'date', 'merchant', 'amount'
            
        Returns:
            Tuple of (has_critical_issues, list_of_all_issues)
            - has_critical_issues: True if transaction should be rejected
            - list_of_all_issues: All validation issues found
        """
        issues = []
        critical = False
        
        # Validate date
        date_valid, date_issue = self.validate_date(transaction.get('date', ''))
        if not date_valid:
            issues.append(f"date:{date_issue}")
            if date_issue == "missing_date":
                critical = True
        
        # Validate merchant
        merchant_valid, merchant_issue = self.validate_merchant(transaction.get('merchant', ''))
        if not merchant_valid:
            issues.append(f"merchant:{merchant_issue}")
            # Missing merchant is not critical - we can mark as "Unknown"
        
        # Validate amount
        amount_valid, amount_issue = self.validate_amount(transaction.get('amount', ''))
        if not amount_valid:
            issues.append(f"amount:{amount_issue}")
            if amount_issue == "missing_amount":
                critical = True
        
        if issues:
            logger.debug(f"Transaction validation issues: {issues}")
        
        return (critical, issues)
    
    def validate_batch(self, transactions: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Validate a batch of transactions.
        
        Args:
            transactions: List of transaction dicts
            
        Returns:
            Dict with validation summary:
            {
                'total': int,
                'valid': int,
                'with_issues': int,
                'critical': int,
                'issues_by_type': dict,
                'critical_indices': list
            }
        """
        total = len(transactions)
        valid_count = 0
        with_issues_count = 0
        critical_count = 0
        critical_indices = []
        issues_by_type = {}
        
        for idx, txn in enumerate(transactions):
            has_critical, issues = self.validate_transaction(txn)
            
            if not issues:
                valid_count += 1
            else:
                with_issues_count += 1
                
                # Count issues by type
                for issue in issues:
                    issues_by_type[issue] = issues_by_type.get(issue, 0) + 1
            
            if has_critical:
                critical_count += 1
                critical_indices.append(idx)
        
        summary = {
            'total': total,
            'valid': valid_count,
            'with_issues': with_issues_count,
            'critical': critical_count,
            'issues_by_type': issues_by_type,
            'critical_indices': critical_indices
        }
        
        logger.info(f"Batch validation: {valid_count}/{total} fully valid, "
                   f"{with_issues_count} with issues, {critical_count} critical")
        
        return summary
    
    def check_completeness(self, transaction: Dict[str, str]) -> float:
        """
        Calculate completeness score for a transaction.
        
        Args:
            transaction: Transaction dict
            
        Returns:
            Float between 0.0 and 1.0 (percentage of fields present)
        """
        required_fields = ['date', 'merchant', 'amount']
        present = 0
        
        for field in required_fields:
            value = transaction.get(field, '')
            if value and value.strip():
                present += 1
        
        return present / len(required_fields)
    
    def identify_data_quality_issues(self, transactions: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Comprehensive data quality analysis.
        
        Args:
            transactions: List of transactions
            
        Returns:
            Dict with quality metrics:
            {
                'completeness_avg': float,
                'missing_dates': int,
                'missing_merchants': int,
                'missing_amounts': int,
                'empty_merchants': int,
                'suspicious_amounts': int
            }
        """
        metrics = {
            'completeness_avg': 0.0,
            'missing_dates': 0,
            'missing_merchants': 0,
            'missing_amounts': 0,
            'empty_merchants': 0,
            'suspicious_amounts': 0
        }
        
        if not transactions:
            return metrics
        
        total_completeness = 0.0
        
        for txn in transactions:
            # Completeness
            total_completeness += self.check_completeness(txn)
            
            # Missing fields
            if not txn.get('date', '').strip():
                metrics['missing_dates'] += 1
            
            merchant = txn.get('merchant', '').strip()
            if not merchant:
                metrics['missing_merchants'] += 1
                metrics['empty_merchants'] += 1
            
            if not txn.get('amount', '').strip():
                metrics['missing_amounts'] += 1
            
            # Suspicious amounts (all zeros, or looks wrong)
            amount = txn.get('amount', '')
            if amount and all(c in '0.$€£ ' for c in amount):
                metrics['suspicious_amounts'] += 1
        
        metrics['completeness_avg'] = total_completeness / len(transactions)
        
        logger.info(f"Data quality: {metrics['completeness_avg']:.1%} complete, "
                   f"{metrics['missing_merchants']} missing merchants, "
                   f"{metrics['missing_amounts']} missing amounts")
        
        return metrics


# Global instance (singleton)
_data_validator_instance = None


def get_data_validator() -> DataValidator:
    """Get global DataValidator instance"""
    global _data_validator_instance
    if _data_validator_instance is None:
        _data_validator_instance = DataValidator()
    return _data_validator_instance
