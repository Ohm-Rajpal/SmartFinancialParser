"""
OpenAI GPT client for merchant normalization.

Implements BaseLLM interface for GPT-5-nano API calls.
"""

import json
import logging
import time
from typing import List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Config
from src.cloud.secrets_manager import get_openai_key
from src.llm.base_llm import BaseLLM, MerchantNormalizationResult
from src.validation.security_validator import get_security_validator

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLM):
    """OpenAI GPT client for merchant normalization using GPT-5-nano"""
    
    def __init__(self):
        """
        Initialize OpenAI client.        
        """

        model = Config.GPT_MODEL
        super().__init__(model)
        
        # Get API key
        api_key = get_openai_key()
        if not api_key:
            raise ValueError("OpenAI API key not found! Set OPENAI_API_KEY in .env")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.security_validator = get_security_validator() # security driven development
        
        logger.info(f"OpenAI client initialized with model: {self.model_name}")
    
    def normalize_merchant(
        self,
        merchant_name: str,
        available_categories: List[str]
    ) -> MerchantNormalizationResult:
        """
        Normalize a merchant name using OpenAI GPT.
        
        Args:
            merchant_name: Raw merchant name (e.g., "UBER *TRIP 12345")
            available_categories: List of valid category names
            
        Returns:
            MerchantNormalizationResult with normalized name and category
        """
        start_time = time.time()
        
        # security driven development, call the helper functions inside of the validator
        sanitized = self.security_validator.sanitize_for_llm(merchant_name)
        
        # Check for CSV injection
        is_safe_csv, reason_csv = self.security_validator.check_csv_injection(merchant_name)
        if not is_safe_csv:
            logger.error(f"Vulnerability detected: {reason_csv}")
            raise ValueError(f"Vulnerability detected: {reason_csv}")
        
        # Check for prompt injection
        is_safe, reason = self.security_validator.check_prompt_injection(sanitized)
        if not is_safe:
            logger.error(f"Vulnerability detected: {reason}")
            raise ValueError(f"Vulnerability detected: {reason}")
        
        # Build prompt
        prompt = self._build_prompt(sanitized, available_categories)
        
        try:
            # Make API call with retry logic
            response = self._call_api_with_retry(prompt)
            
            # Parse response
            result = self._parse_response(response, merchant_name, start_time, available_categories)
            
            # Update stats
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
            self._update_stats(tokens_used)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to normalize merchant '{merchant_name}': {e}")
            # Return a fallback result
            processing_time = time.time() - start_time
            return MerchantNormalizationResult(
                original_name=merchant_name,
                normalized_name=merchant_name,  # Fallback: use original
                category="Other",  # Fallback category
                confidence=0.0,
                model_name=self.model_name,
                processing_time=processing_time,
                reasoning=f"Error: {str(e)}"
            )
    
    # ensures we can handle API call failures
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_api_with_retry(self, prompt: str):
        """
        Make API call with automatic retry on failure.
        
        Args:
            prompt: The prompt to send. Should be the same as the base_llm!
            
        Returns:
            OpenAI response object
        """
        try:
            # gpt-5-nano-2025-08-07 requires max_completion_tokens and doesn't support temperature=0
            # It only supports the default temperature (1), so we omit the temperature parameter
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial transaction categorization expert. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                # temperature=0 not supported by gpt-5-nano, uses default (1) instead
                max_completion_tokens=Config.MAX_TOKENS,  # gpt-5-nano uses max_completion_tokens
                timeout=Config.REQUEST_TIMEOUT,
                response_format={"type": "json_object"}  # json
            )
            return response
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}, retrying...")
            raise
    
    
    def _validate_and_fix_category(
        self,
        category: str,
        available_categories: List[str],
        merchant_name: str
    ) -> str:
        """
        Validate category and fix if invalid.
        
        Args:
            category: Category from LLM
            available_categories: Valid categories
            merchant_name: Original merchant (for fallback)
            
        Returns:
            Valid category
        """
        # Exact match
        if category in available_categories:
            return category
        
        # Fuzzy match (case-insensitive, strip spaces)
        category_lower = category.lower().strip()
        for valid in available_categories:
            if category_lower == valid.lower().strip():
                logger.info(f"Fuzzy matched '{category}' -> '{valid}'")
                return valid
        
        # Substring match
        for valid in available_categories:
            if category_lower in valid.lower() or valid.lower() in category_lower:
                logger.info(f"Substring matched '{category}' -> '{valid}'")
                return valid
        
        # Keyword fallback
        logger.warning(f"Category '{category}' not recognized, using keyword fallback")
        return self._keyword_fallback(merchant_name)

    def _keyword_fallback(self, merchant_name: str) -> str:
        """Use Config.CATEGORY_KEYWORDS as fallback"""
        merchant_lower = merchant_name.lower()
        
        for category, keywords in Config.CATEGORY_KEYWORDS.items():
            if any(keyword in merchant_lower for keyword in keywords):
                logger.info(f"Keyword fallback: '{merchant_name}' -> '{category}'")
                return category
        
        return "Other"

    def _parse_response(
        self,
        response,
        original_merchant: str,
        start_time: float,
        available_categories: List[str]
    ) -> MerchantNormalizationResult:
        """
        Parse OpenAI API response into MerchantNormalizationResult.
        
        Args:
            response: OpenAI response object
            original_merchant: Original merchant name
            start_time: Start time for processing_time calculation
            
        Returns:
            MerchantNormalizationResult
        """
        processing_time = time.time() - start_time
        
        # Extract content
        content = response.choices[0].message.content.strip()
        
        # Parse JSON (handle markdown code blocks if present)
        content = content.replace("```json", "").replace("```", "").strip()
        
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {content[:200]}")
            raise ValueError(f"Invalid JSON response: {e}")
        
        # Validate required fields
        normalized_name = parsed.get("normalized_name", original_merchant)
        category = parsed.get("category", "Other")
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = parsed.get("reasoning")  # this is for debugging purposes right now
        
        # Validate and fix category
        category = self._validate_and_fix_category(
            category,
            available_categories,
            original_merchant
        )
        
        # Restrict confidence to prevent faulty behavior due to hallucinations
        confidence = max(0.0, min(1.0, confidence))
        
        return MerchantNormalizationResult(
            original_name=original_merchant,
            normalized_name=normalized_name,
            category=category,
            confidence=confidence,
            model_name=self.model_name,
            processing_time=processing_time,
            reasoning=reasoning
        )
    
    def normalize_batch(
        self,
        merchant_names: List[str],
        available_categories: List[str]
    ) -> List[MerchantNormalizationResult]:
        """
        Normalize a batch of merchant names.
        
        Args:
            merchant_names: List of merchant names
            available_categories: List of valid categories
            
        Returns:
            List of MerchantNormalizationResult
        """
        results = []
        for merchant_name in merchant_names:
            try:
                result = self.normalize_merchant(merchant_name, available_categories)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to normalize '{merchant_name}' in batch: {e}")
                # Add fallback result
                results.append(MerchantNormalizationResult(
                    original_name=merchant_name,
                    normalized_name=merchant_name,
                    category="Other",
                    confidence=0.0,
                    model_name=self.model_name,
                    processing_time=0.0,
                    reasoning=f"Error: {str(e)}"
                ))
        
        return results