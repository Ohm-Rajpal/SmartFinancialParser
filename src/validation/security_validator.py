"""
Security validation to prevent injection

Validates user input and data to prevent:
- CSV injections
- Prompt injection
- Path traversal attacks

All data must pass security validation before processing.
"""

import re
import logging
from typing import List, Tuple, Optional

from src.config import Config

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Validates data for security vulnerabilities"""
    
    def __init__(self):
        """Initialize security validator with patterns from config"""
        self.csv_injection_prefixes = Config.CSV_INJECTION_PREFIXES
        self.prompt_injection_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in Config.PROMPT_INJECTION_PATTERNS
        ]
        self.path_traversal_patterns = [
            re.compile(pattern) 
            for pattern in Config.PATH_TRAVERSAL_PATTERNS
        ]
    
    def check_csv_injection(self, value: str) -> Tuple[bool, Optional[str]]:
        """
        Check if value contains CSV injection attempt.
        
        CSV injection occurs when cells start with =, +, -, @, etc.
        Excel/Sheets will execute these as formulas.
        
        Args:
            value: String to check
            
        Returns:
            Tuple of (is_safe, reason)
            - (True, None) if safe
            - (False, "reason") if dangerous
            
        Example:
            >>> validator = SecurityValidator()
            >>> validator.check_csv_injection("=1+1")
            (False, "CSV injection: starts with dangerous character '='")
        """
        if not value or not isinstance(value, str):
            return (True, None)
        
        # Check if starts with dangerous character
        if value and value[0] in self.csv_injection_prefixes:
            char = value[0]
            logger.warning(f"CSV injection detected: '{value[:50]}'")
            return (False, f"CSV injection: starts with dangerous character '{char}'")
        
        return (True, None)
    
    def check_prompt_injection(self, value: str) -> Tuple[bool, Optional[str]]:
        """
        Check if value contains prompt injection attempt.
        
        Prompt injection tries to manipulate LLM behavior with commands like:
        - "ignore previous instructions"
        - "system: you are now..."
        - "disregard all rules"
        
        Args:
            value: String to check
            
        Returns:
            Tuple of (is_safe, reason)
        """
        if not value or not isinstance(value, str):
            return (True, None)
        
        # Check against known patterns
        for pattern in self.prompt_injection_patterns:
            if pattern.search(value):
                logger.warning(f"Prompt injection detected: '{value[:50]}'")
                return (False, f"Prompt injection: matches pattern '{pattern.pattern}'")
        
        return (True, None)
    
    def check_path_traversal(self, path: str) -> Tuple[bool, Optional[str]]:
        """
        Check if path contains traversal attempt.
        
        Path traversal tries to access files outside allowed directories:
        - ../../etc/passwd
        - /root/
        - ~/
        
        Args:
            path: File path to check
            
        Returns:
            Tuple of (is_safe, reason)
        """
        if not path or not isinstance(path, str):
            return (True, None)
        
        # Check against traversal patterns
        for pattern in self.path_traversal_patterns:
            if pattern.search(path):
                logger.warning(f"Path traversal detected: '{path}'")
                return (False, f"Path traversal: matches pattern '{pattern.pattern}'")
        
        return (True, None)
    
    def sanitize_csv_value(self, value: str) -> str:
        """
        Sanitize a value for safe CSV output.
        
        If value starts with dangerous character, prefix with single quote
        to force Excel/Sheets to treat as text.
        
        Args:
            value: Value to sanitize
            
        Returns:
            Sanitized value safe for CSV
            
        Example:
            >>> validator = SecurityValidator()
            >>> validator.sanitize_csv_value("=1+1")
            "'=1+1"
        """
        if not value or not isinstance(value, str):
            return value
        
        # If starts with dangerous char, escape it
        if value and value[0] in self.csv_injection_prefixes:
            sanitized = "'" + value
            logger.info(f"Sanitized CSV value: '{value[:30]}' -> '{sanitized[:30]}'")
            return sanitized
        
        return value
    
    def sanitize_for_llm(self, value: str) -> str:
        """
        Sanitize value before sending to LLM.
        
        Removes or escapes potentially dangerous content while preserving
        the essential information for normalization.
        
        Args:
            value: Value to sanitize
            
        Returns:
            Sanitized value safe for LLM
        """
        if not value or not isinstance(value, str):
            return value
        
        # Strip leading/trailing whitespace
        cleaned = value.strip()
        
        # Remove null bytes
        cleaned = cleaned.replace('\x00', '')
        
        # Limit length to prevent token overflow
        max_length = 500
        if len(cleaned) > max_length:
            logger.warning(f"Truncating long value: {len(cleaned)} -> {max_length} chars")
            cleaned = cleaned[:max_length]
        
        return cleaned
    
    def validate_merchant_name(self, merchant: str) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of merchant name.
        
        Checks for:
        - CSV injection
        - Prompt injection
        - Length limits
        
        Args:
            merchant: Merchant name to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if empty
        if not merchant or not merchant.strip():
            issues.append("Empty merchant name")
            return (False, issues)
        
        # Check length
        if len(merchant) > Config.MAX_MERCHANT_NAME_LENGTH:
            issues.append(f"Merchant name too long ({len(merchant)} > {Config.MAX_MERCHANT_NAME_LENGTH})")
        
        if len(merchant.strip()) < Config.MIN_MERCHANT_NAME_LENGTH:
            issues.append("Merchant name too short")
        
        # Security checks
        csv_safe, csv_reason = self.check_csv_injection(merchant)
        if not csv_safe:
            issues.append(csv_reason)
        
        prompt_safe, prompt_reason = self.check_prompt_injection(merchant)
        if not prompt_safe:
            issues.append(prompt_reason)
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"Merchant validation failed: '{merchant[:50]}' - {issues}")
        
        return (is_valid, issues)
    
    def validate_file_path(self, path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate file path for security.
        
        Ensures path does not contain traversal attempts.
        
        Args:
            path: File path to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not path:
            return (False, "Empty path")
        
        # Check path traversal
        safe, reason = self.check_path_traversal(path)
        if not safe:
            return (False, reason)
        
        return (True, None)
    
    def validate_batch(self, merchants: List[str]) -> Tuple[List[str], List[Tuple[int, str, List[str]]]]:
        """
        Validate a batch of merchant names.
        
        Args:
            merchants: List of merchant names
            
        Returns:
            Tuple of (valid_merchants, invalid_entries)
            - valid_merchants: List of merchants that passed validation
            - invalid_entries: List of (index, merchant, issues)
        """
        valid = []
        invalid = []
        
        for idx, merchant in enumerate(merchants):
            is_valid, issues = self.validate_merchant_name(merchant)
            
            if is_valid:
                valid.append(merchant)
            else:
                invalid.append((idx, merchant, issues))
        
        if invalid:
            logger.warning(f"Batch validation: {len(invalid)}/{len(merchants)} merchants failed")
        
        return (valid, invalid)


# Global instance (singleton)
_security_validator_instance = None


def get_security_validator() -> SecurityValidator:
    """Get global SecurityValidator instance"""
    global _security_validator_instance
    if _security_validator_instance is None:
        _security_validator_instance = SecurityValidator()
    return _security_validator_instance
