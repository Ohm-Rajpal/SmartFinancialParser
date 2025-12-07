"""
Base LLM interface for merchant normalization.

Defines abstract interface that all LLM clients must implement.
Ensures consistent API across the two models this project uses (GPT and Gemini)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass
import logging
import threading # Added threading library due to parallel execution
import time

logger = logging.getLogger(__name__)


@dataclass
class MerchantNormalizationResult:
    """Result from LLM merchant normalization"""
    original_name: str
    normalized_name: str # ai powered analysis of the original name
    category: str # this is the final analysis criteria so it's very important to get right
    confidence: float  # 0.0 to 1.0, used for tie breaking
    model_name: str
    processing_time: float  # seconds
    reasoning: Optional[str] = None  # Explain categorization, do not need to store in S3 necessarily though


class BaseLLM(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, model_name: str):
        """
        Initialize LLM client.
        
        Args:
            model_name: Model we use
        """
        self.model_name = model_name
        self.temperature = 0 # hardcoded to 0 for maximum determinism
        self._lock = threading.Lock() # Thread-safe counters for Ray parallel execution
        self.request_count = 0
        self.total_tokens = 0
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def normalize_merchant(self, merchant_name: str, available_categories: list) -> MerchantNormalizationResult:
        """
        Normalize a merchant name and categorize it. Critical for final analysis.
        
        Args:
            merchant_name: Raw merchant name (e.g., "UBER *TRIP 12345")
            available_categories: List of valid category names
            
        Returns:
            MerchantNormalizationResult with normalized name and category.
        """
        pass
    
    @abstractmethod
    def normalize_batch(self, merchant_names: list, available_categories: list) -> list:
        """
        Normalize a batch of merchant names.
        
        Some LLMs support batch processing for efficiency.
        
        Args:
            merchant_names: List of merchant names
            available_categories: List of valid categories
            
        Returns:
            List of MerchantNormalizationResult
        """
        pass
    
    def _build_prompt(self, merchant_name: str, available_categories: list) -> str:
        """
        Build normalization prompt for LLM with strict instructions and examples.
        
        Args:
            merchant_name: Merchant name to normalize
            available_categories: Available categories
            
        Returns:
            Formatted prompt string with few-shot learning technique
        """
        # Format categories as numbered list for clarity
        categories_str = "\n   ".join(f"{i+1}. {cat}" for i, cat in enumerate(available_categories))
        
        prompt = f"""You are a financial transaction categorization expert. Your task is to normalize merchant names and assign categories.

            Merchant name: "{merchant_name}"

            STRICT REQUIREMENTS:
            1. Normalize to the OFFICIAL brand name (e.g., "STARBUCKS #1234" → "Starbucks", not "starbucks" or "STARBUCKS")
            - Use proper capitalization (Starbucks, Amazon, Uber)
            - Remove transaction codes, location numbers, asterisks
            - Keep only the core brand name
            2. Choose EXACTLY ONE category from this list (DO NOT create new categories):
            {categories_str}
            3. If uncertain, use "Other" - NEVER invent categories
            4. Provide confidence score (0.0-1.0) based on certainty

            EXAMPLES OF CORRECT NORMALIZATION:
            {{"normalized_name": "Starbucks", "category": "Food & Dining", "confidence": 0.95, "reasoning": "Coffee shop chain"}}
            {{"normalized_name": "Uber", "category": "Transportation", "confidence": 0.90, "reasoning": "Ride-sharing service indicator *TRIP"}}
            {{"normalized_name": "Amazon", "category": "Shopping", "confidence": 0.85, "reasoning": "Online marketplace"}}
            {{"normalized_name": "Netflix", "category": "Entertainment", "confidence": 0.95, "reasoning": "Streaming service"}}
            {{"normalized_name": "Shell", "category": "Transportation", "confidence": 0.90, "reasoning": "Gas station"}}

            NORMALIZATION RULES:
            - "UBER *TRIP 12345" → "Uber" (not "UBER" or "uber")
            - "amazon.com*123ABC" → "Amazon" (not "Amazon.com")
            - "MCDONALDS #5678" → "McDonald's" (not "MCDONALDS" or "McDonalds")
            - "WALGREENS STORE" → "Walgreens" (not "WALGREENS STORE")

            Respond ONLY with valid JSON (no markdown, no code blocks, no extra text):
            {{
                "normalized_name": "Proper Brand Name",
                "category": "MUST be from numbered list above",
                "confidence": 0.0-1.0,
                "reasoning": "Brief explanation"
            }}"""
        
        return prompt
    
    def _update_stats(self, tokens_used: int):
        """
        Thread-safe stats update for Ray parallel execution.
        
        Args:
            tokens_used: Number of tokens used in this request
        """
        with self._lock:
            self.request_count += 1
            self.total_tokens += tokens_used
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get usage statistics (thread-safe).
        
        Returns:
            Dict with request_count, total_tokens, etc. (for this worker only)
        """
        with self._lock:
            return {
                'model_name': self.model_name,
                'request_count': self.request_count,
                'total_tokens': self.total_tokens,
                'avg_tokens_per_request': self.total_tokens / max(self.request_count, 1)
            }
    
    @staticmethod
    def aggregate_stats(stats_list: list[Dict]) -> Dict[str, any]:
        """
        Aggregate stats across multiple Ray workers.
        
        Use this in model_comparator.py to combine stats from all workers. We need to ensure
        that we call this static method on both openai and gemini clients!
        
        Args:
            stats_list: List of stats dicts from multiple workers
            
        Returns:
            Aggregated stats dict
        """
        if not stats_list:
            return {}
        
        total_requests = sum(s.get('request_count', 0) for s in stats_list)
        total_tokens = sum(s.get('total_tokens', 0) for s in stats_list)
        model_name = stats_list[0].get('model_name', 'unknown client, please investigate')
        
        return {
            'model_name': model_name,
            'total_request_count': total_requests,
            'total_tokens': total_tokens,
            'num_workers': len(stats_list),
            'avg_tokens_per_request': total_tokens / max(total_requests, 1),
            'per_worker_stats': stats_list
            # note to self: add more parameters later to check if Ray 
            # is distributing tasks fairly
        }