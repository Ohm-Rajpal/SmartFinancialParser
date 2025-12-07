"""
Date normalization module for normalizing date formats.

Normalizes date formats to a consistent format.
"""

import logging
import ciso8601
import dateparser

from typing import Dict, List, Tuple, Optional

from src.config import Config

logger = logging.getLogger(__name__)

class DateNormalizer:
    """Normalizes date formats to a consistent format"""
    
    def __init__(self, is_valid_date_format: bool = True):
        """Initialize date normalizer with config rules"""
        self.is_valid_date_format = is_valid_date_format
        
    def normalize_date(self, date_str: str) -> str:
        """
        Normalize date format to a consistent format.
        """

        if not self.is_valid_date_format or not isinstance(date_str, str):
            return None

        try:
            # Fast path for ISO / numeric formats
            return ciso8601.parse_datetime(date_str).date().isoformat()
        except:
            # Flexible fallback
            dt = dateparser.parse(date_str)
            if dt:
                return dt.date().isoformat()
            return None
