"""
AWS Secrets Manager integration for secure API key retrieval.

This module fetches API keys and secrets from AWS Secrets Manager,
with fallback to local environment variables for development. In this hackathon,
I decided to use the local environment variables for speeding up development
"""

import json
import logging
from typing import Dict, Optional

from src.config import Config

# Setup logging
logger = logging.getLogger(__name__)


class SecretsManager:
    """Secure retrieval of API keys and secrets"""
    
    def __init__(self):
        """Initialize Secrets Manager"""
        self.region = Config.AWS_REGION
        
    def get_openai_api_key(self) -> Optional[str]:
        """
        Get OpenAI API key from local environment.
        
        Returns:
            OpenAI API key string, or None if not found
        """
        if Config.OPENAI_API_KEY:
            logger.info("OpenAI API key loaded from .env")
            return Config.OPENAI_API_KEY
        
        logger.error("OpenAI API key not found in .env!")
        return None
    
    def get_gemini_api_key(self) -> Optional[str]:
        """
        Get Google Gemini API key from local environment.
        
        Returns:
            Gemini API key string, or None if not found
        """
        if Config.GEMINI_API_KEY:
            logger.info("Gemini API key loaded from .env")
            return Config.GEMINI_API_KEY
        
        logger.error("Gemini API key not found in .env!")
        return None
    
    def get_encryption_key(self) -> Optional[str]:
        """
        Get encryption key for local data encryption.
        
        Returns:
            Encryption key string, or None (optional for hackathon)
        """
        logger.info("Encryption key skipped for hackathon (optional feature)")
        return None
    
    def validate_all_secrets(self) -> bool:
        """
        Validate that all required secrets are accessible.
        
        Returns:
            bool: True if all secrets are available, False otherwise
        """
        logger.info("Validating all required secrets...")
        
        all_valid = True
        
        # Check OpenAI key
        openai_key = self.get_openai_api_key()
        if openai_key:
            logger.info("OpenAI API key available")
        else:
            logger.error("OpenAI API key NOT available")
            all_valid = False
        
        # Check Gemini key
        gemini_key = self.get_gemini_api_key()
        if gemini_key:
            logger.info("Gemini API key available")
        else:
            logger.error("Gemini API key NOT available")
            all_valid = False
        
        # Encryption key is optional
        logger.info("Encryption key skipped (optional for hackathon)")
        
        if all_valid:
            logger.info("All required API keys validated successfully!")
        
        return all_valid


# Global instance (singleton pattern)
_secrets_manager_instance = None


def get_secrets_manager() -> SecretsManager:
    """
    Get global SecretsManager instance (singleton).
    
    Returns:
        SecretsManager instance
    """
    global _secrets_manager_instance
    if _secrets_manager_instance is None:
        _secrets_manager_instance = SecretsManager()
    return _secrets_manager_instance


# Convenience functions
def get_openai_key() -> Optional[str]:
    """Convenience function to get OpenAI API key"""
    return get_secrets_manager().get_openai_api_key()


def get_gemini_key() -> Optional[str]:
    """Convenience function to get Gemini API key"""
    return get_secrets_manager().get_gemini_api_key()


def validate_secrets() -> bool:
    """Convenience function to validate all secrets"""
    return get_secrets_manager().validate_all_secrets()
