"""
Google Gemini client for merchant normalization.

Implements BaseLLM interface for Gemini API calls.
"""

import json
import logging
import time
from typing import List, Optional

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Config
from src.cloud.secrets_manager import get_gemini_key
from src.llm.base_llm import BaseLLM, MerchantNormalizationResult
from src.validation.security_validator import get_security_validator

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLM):
    """Google Gemini client for merchant normalization"""
    
    def __init__(self):
        """
        Initialize Gemini client.
        """
        model = Config.GEMINI_MODEL
        super().__init__(model)
        
        # Get API key
        api_key = get_gemini_key()
        if not api_key:
            raise ValueError("Gemini API key not found! Set GEMINI_API_KEY in .env")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": Config.MAX_TOKENS,
            }
        )
        self.security_validator = get_security_validator() # security driven development
        
        logger.info(f"Gemini client initialized with model: {self.model_name}")
    
    def normalize_merchant(
        self,
        merchant_name: str,
        available_categories: List[str]
    ) -> MerchantNormalizationResult:
        """
        Normalize a merchant name using Google Gemini.
        
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
            
            # Parse response (includes category validation)
            result = self._parse_response(response, merchant_name, start_time, available_categories)
            
            # Update stats (estimation)
            # 1 token is approximately 4 characters
            tokens_used = len(prompt) // 4 + len(result.normalized_name) // 4
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
    
    # this is to ensure we do not fail too easily if the API call fails
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
            Gemini response object
        """
        try:
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": Config.MAX_TOKENS,
                }
            )
            return response
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}, retrying...")
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
        if category in available_categories:
            return category
        
        # slower: fuzzy matching
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
        Parse Gemini API response into MerchantNormalizationResult.
        
        Args:
            response: Gemini response object
            original_merchant: Original merchant name
            start_time: Start time for processing_time calculation
            
        Returns:
            MerchantNormalizationResult
        """
        processing_time = time.time() - start_time # elapsed time
        
        # Extract content
        content = response.text.strip()
        
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
        reasoning = parsed.get("reasoning")
        
        # Validate and fix category (fuzzy matching, keyword fallback)
        category = self._validate_and_fix_category(
            category,
            available_categories,
            original_merchant
        )
        
        # Restrict confidence to 0.0-1.0 in case hallucination occurs
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