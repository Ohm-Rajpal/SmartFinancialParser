"""
Comprehensive test suite for LLM clients with edge case coverage.
"""

import pytest
from unittest.mock import Mock, patch
from src.llm.openai_client import OpenAIClient
from src.llm.gemini_client import GeminiClient
from src.llm.base_llm import MerchantNormalizationResult
from src.config import Config


class TestOpenAIClientComprehensive:
    """Comprehensive OpenAI client tests"""
    
    @pytest.fixture
    def client(self):
        """Create mocked OpenAI client"""
        with patch('src.llm.openai_client.get_openai_key', return_value='test-key'):
            with patch('src.llm.openai_client.OpenAI'):
                return OpenAIClient()

    def test_invalid_category_gets_fixed(self, client):
        """Test that invalid LLM categories are corrected"""
        # Mock response with INVALID category
        mock_response = Mock()
        mock_response.choices = [Mock()]
        # LLM returns "Food and Dining" (wrong - should be "Food & Dining")
        mock_response.choices[0].message.content = '{"normalized_name": "Starbucks", "category": "Food and Dining", "confidence": 0.95, "reasoning": "Coffee"}'
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 100
        
        with patch.object(client.client.chat.completions, 'create', return_value=mock_response):
            result = client.normalize_merchant("STARBUCKS", Config.MERCHANT_CATEGORIES)
            
            # Should auto-correct to valid category
            assert result.category == "Food & Dining"  # Not "Food and Dining"
            assert result.category in Config.MERCHANT_CATEGORIES

    def test_completely_invalid_category_uses_fallback(self, client):
        """Test that garbage categories fall back to keywords"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        # LLM hallucinates a category
        mock_response.choices[0].message.content = '{"normalized_name": "Starbucks", "category": "CoffeePlaces", "confidence": 0.95, "reasoning": "Test"}'
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 100
        
        with patch.object(client.client.chat.completions, 'create', return_value=mock_response):
            result = client.normalize_merchant("STARBUCKS COFFEE", Config.MERCHANT_CATEGORIES)
            
            # Should use keyword fallback since "CoffeePlaces" is invalid
            assert result.category == "Food & Dining"  # From keyword "coffee"
            assert result.category in Config.MERCHANT_CATEGORIES

    def test_malformed_json_response(self, client):
        """Test handling of malformed JSON from LLM"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        # Invalid JSON
        mock_response.choices[0].message.content = '{invalid json here'
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 50
        
        with patch.object(client.client.chat.completions, 'create', return_value=mock_response):
            result = client.normalize_merchant("TEST MERCHANT", Config.MERCHANT_CATEGORIES)
            
            # Should return fallback result
            assert result.original_name == "TEST MERCHANT"
            assert result.category in Config.MERCHANT_CATEGORIES  # Valid category
            assert result.confidence == 0.0  # Low confidence for error

    def test_markdown_code_blocks_stripped(self, client):
        """Test that markdown code blocks are properly handled"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        # JSON wrapped in markdown
        mock_response.choices[0].message.content = '```json\n{"normalized_name": "Uber", "category": "Transportation", "confidence": 0.90, "reasoning": "Ride"}\n```'
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 100
        
        with patch.object(client.client.chat.completions, 'create', return_value=mock_response):
            result = client.normalize_merchant("UBER", Config.MERCHANT_CATEGORIES)
            
            # Should successfully parse despite markdown
            assert result.normalized_name == "Uber"
            assert result.category == "Transportation"

    def test_api_timeout_returns_fallback(self, client):
        """Test handling of API timeout"""
        with patch.object(client.client.chat.completions, 'create', side_effect=TimeoutError("API timeout")):
            result = client.normalize_merchant("TEST", Config.MERCHANT_CATEGORIES)
            
            # Should return fallback, not crash
            assert result.original_name == "TEST"
            assert result.category in Config.MERCHANT_CATEGORIES
            assert "Error" in result.reasoning or "timeout" in result.reasoning.lower()

    def test_api_rate_limit_retries(self, client):
        """Test retry logic on rate limit"""
        # First call fails, second succeeds
        mock_success = Mock()
        mock_success.choices = [Mock()]
        mock_success.choices[0].message.content = '{"normalized_name": "Test", "category": "Other", "confidence": 0.5, "reasoning": "Test"}'
        mock_success.usage = Mock()
        mock_success.usage.total_tokens = 50
        
        with patch.object(
            client.client.chat.completions,
            'create',
            side_effect=[Exception("Rate limit"), mock_success]  # Fail then succeed
        ):
            result = client.normalize_merchant("TEST", Config.MERCHANT_CATEGORIES)
            
            # Should succeed after retry
            assert result.normalized_name == "Test"
            assert result.category == "Other"

    def test_confidence_clamped_to_valid_range(self, client):
        """Test that invalid confidence scores are clamped (must be between 0 and 1.0)"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        # LLM returns confidence > 1.0
        mock_response.choices[0].message.content = '{"normalized_name": "Test", "category": "Other", "confidence": 1.5, "reasoning": "Test"}'
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 50
        
        with patch.object(client.client.chat.completions, 'create', return_value=mock_response):
            result = client.normalize_merchant("TEST", Config.MERCHANT_CATEGORIES)
            
            # Should clamp to 1.0
            assert result.confidence == 1.0
            assert 0.0 <= result.confidence <= 1.0

    def test_negative_confidence_clamped_to_zero(self, client):
        """Test negative confidence is clamped to 0"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"normalized_name": "Test", "category": "Other", "confidence": -0.5, "reasoning": "Test"}'
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 50
        
        with patch.object(client.client.chat.completions, 'create', return_value=mock_response):
            result = client.normalize_merchant("TEST", Config.MERCHANT_CATEGORIES)
            
            assert result.confidence == 0.0

    def test_normalize_batch_processes_all(self, client):
        """Test batch processing handles multiple merchants"""
        merchants = ["UBER", "STARBUCKS", "AMAZON"]
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"normalized_name": "Test", "category": "Other", "confidence": 0.5, "reasoning": "Test"}'
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 50
        
        with patch.object(client.client.chat.completions, 'create', return_value=mock_response):
            results = client.normalize_batch(merchants, Config.MERCHANT_CATEGORIES)
            
            # Should return result for each merchant
            assert len(results) == 3
            assert all(isinstance(r, MerchantNormalizationResult) for r in results)

    def test_normalize_batch_continues_on_error(self, client):
        """Test batch processing continues even if one merchant fails"""
        merchants = ["UBER", "INVALID", "AMAZON"]
        
        mock_success = Mock()
        mock_success.choices = [Mock()]
        mock_success.choices[0].message.content = '{"normalized_name": "Test", "category": "Other", "confidence": 0.5, "reasoning": "Test"}'
        mock_success.usage = Mock()
        mock_success.usage.total_tokens = 50
        
        # First and third succeed, middle fails (with retries: 3 attempts per failure)
        # Retry decorator will try 3 times, so we need 3 exceptions for the middle one
        with patch.object(
            client.client.chat.completions,
            'create',
            side_effect=[
                mock_success,  # UBER succeeds
                Exception("API error"),  # INVALID fails attempt 1
                Exception("API error"),  # INVALID fails attempt 2
                Exception("API error"),  # INVALID fails attempt 3
                mock_success  # AMAZON succeeds
            ]
        ):
            results = client.normalize_batch(merchants, Config.MERCHANT_CATEGORIES)
            
            # Should still return 3 results (with fallback for failed one)
            assert len(results) == 3
            assert results[1].confidence == 0.0  # Failed merchant has low confidence
            assert results[1].original_name == "INVALID"
        
    def test_csv_injection_sanitized(self, client):
        """Test that CSV injection attempts are sanitized"""
        # security validator should catch this
        with pytest.raises(ValueError, match="Vulnerability detected"):
            client.normalize_merchant("=1+1", Config.MERCHANT_CATEGORIES)

    def test_merchant_name_too_long_truncated(self, client):
        """Test very long merchant names are handled"""
        long_merchant = "A" * 600  # Exceeds MAX length
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"normalized_name": "Test", "category": "Other", "confidence": 0.5, "reasoning": "Test"}'
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 50
        
        with patch.object(client.client.chat.completions, 'create', return_value=mock_response):
            # should not crash because of truncation
            result = client.normalize_merchant(long_merchant, Config.MERCHANT_CATEGORIES)
            assert result is not None
    
    def test_stats_updated_after_requests(self, client):
        """Test that request stats are tracked correctly"""
        initial_count = client.request_count
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"normalized_name": "Test", "category": "Other", "confidence": 0.5, "reasoning": "Test"}'
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 150
        
        with patch.object(client.client.chat.completions, 'create', return_value=mock_response):
            client.normalize_merchant("TEST", Config.MERCHANT_CATEGORIES)
            
            # Stats should be updated
            assert client.request_count == initial_count + 1
            assert client.total_tokens >= 150

    def test_get_stats_returns_correct_format(self, client):
        """Test stats dictionary format"""
        stats = client.get_stats()
        
        assert 'model_name' in stats
        assert 'request_count' in stats
        assert 'total_tokens' in stats
        assert 'avg_tokens_per_request' in stats
        assert isinstance(stats['request_count'], int)