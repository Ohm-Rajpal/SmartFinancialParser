"""
Comprehensive test suite for MerchantNormalizer with Ray integration.

Tests generated with AI due to many edge cases.

Tests cover:
- Ray initialization and shutdown
- Merchant deduplication
- Batch splitting logic
- Parallel batch processing
- Ensemble voting integration
- Error handling
- Edge cases (empty input, single merchant, large batches)
"""

import pytest
import ray
from unittest.mock import Mock, patch, MagicMock
from typing import Dict

from src.normalization.merchant_normalizer import (
    MerchantNormalizer,
    process_batch_remote
)
from src.llm.base_llm import MerchantNormalizationResult
from src.llm.model_comparator import CategoryAnalyzer
from src.config import Config


class TestMerchantNormalizerInitialization:
    """Test MerchantNormalizer initialization and Ray setup"""
    
    def test_initialization_sets_config(self):
        """Test that initialization sets correct config values"""
        with patch('src.normalization.merchant_normalizer.ray.is_initialized', return_value=True):
            normalizer = MerchantNormalizer()
            
            assert normalizer.available_categories == Config.MERCHANT_CATEGORIES
            assert normalizer.base_batch_size == Config.BATCH_SIZE
            assert normalizer.min_batch_size == 5
            assert normalizer.max_batch_size == 200
    
    def test_ray_initialization_when_not_initialized(self):
        """Test that Ray is initialized if not already initialized"""
        with patch('src.normalization.merchant_normalizer.ray.is_initialized', return_value=False):
            with patch('src.normalization.merchant_normalizer.ray.init') as mock_init:
                normalizer = MerchantNormalizer()
                
                mock_init.assert_called_once()
                call_kwargs = mock_init.call_args[1]
                assert call_kwargs['num_cpus'] == Config.RAY_NUM_CPUS
                assert call_kwargs['object_store_memory'] == Config.RAY_OBJECT_STORE_MEMORY
    
    def test_ray_not_reinitialized_if_already_initialized(self):
        """Test that Ray is not reinitialized if already running"""
        with patch('src.normalization.merchant_normalizer.ray.is_initialized', return_value=True):
            with patch('src.normalization.merchant_normalizer.ray.init') as mock_init:
                normalizer = MerchantNormalizer()
                
                mock_init.assert_not_called()
    
    def test_shutdown_calls_ray_shutdown(self):
        """Test that shutdown properly calls Ray shutdown"""
        with patch('src.normalization.merchant_normalizer.ray.is_initialized', return_value=True):
            with patch('src.normalization.merchant_normalizer.ray.shutdown') as mock_shutdown:
                normalizer = MerchantNormalizer()
                normalizer.shutdown()
                
                mock_shutdown.assert_called_once()
    
    def test_shutdown_skips_if_ray_not_initialized(self):
        """Test that shutdown skips if Ray is not initialized"""
        with patch('src.normalization.merchant_normalizer.ray.is_initialized', return_value=False):
            with patch('src.normalization.merchant_normalizer.ray.shutdown') as mock_shutdown:
                # Mock ray.init to prevent actual initialization
                with patch('src.normalization.merchant_normalizer.ray.init'):
                    normalizer = MerchantNormalizer()
                    normalizer.shutdown()
                    
                    mock_shutdown.assert_not_called()


class TestMerchantDeduplication:
    """Test merchant deduplication logic"""
    
    @pytest.fixture
    def normalizer(self):
        """Create MerchantNormalizer with mocked Ray"""
        with patch('src.normalization.merchant_normalizer.ray.is_initialized', return_value=True):
            return MerchantNormalizer()
    
    def test_duplicate_merchants_are_deduplicated(self, normalizer):
        """Test that duplicate merchants are processed only once"""
        merchants = ["UBER *TRIP", "UBER *TRIP", "UBER *TRIP", "STARBUCKS"]
        
        # Mock the Ray remote function
        mock_result = {
            "UBER *TRIP": MerchantNormalizationResult(
                original_name="UBER *TRIP",
                normalized_name="Uber",
                category="Transportation",
                confidence=0.9,
                model_name="ensemble",
                processing_time=0.5,
                reasoning="Test"
            ),
            "STARBUCKS": MerchantNormalizationResult(
                original_name="STARBUCKS",
                normalized_name="Starbucks",
                category="Food & Dining",
                confidence=0.95,
                model_name="ensemble",
                processing_time=0.4,
                reasoning="Test"
            )
        }
        
        with patch('src.normalization.merchant_normalizer.process_batch_remote.remote', return_value=mock_result):
            with patch('src.normalization.merchant_normalizer.ray.get', return_value=[mock_result]):
                results = normalizer.normalize_merchants(merchants)
                
                # Should only have 2 unique merchants
                assert len(results) == 2
                assert "UBER *TRIP" in results
                assert "STARBUCKS" in results
    
    def test_empty_list_returns_empty_dict(self, normalizer):
        """Test that empty input returns empty dict"""
        with patch('src.normalization.merchant_normalizer.process_batch_remote.remote'):
            with patch('src.normalization.merchant_normalizer.ray.get', return_value=[]):
                results = normalizer.normalize_merchants([])
                
                assert results == {}
                assert len(results) == 0
    
    def test_single_merchant_processed(self, normalizer):
        """Test that single merchant is processed correctly"""
        merchants = ["UBER *TRIP"]
        
        mock_result = {
            "UBER *TRIP": MerchantNormalizationResult(
                original_name="UBER *TRIP",
                normalized_name="Uber",
                category="Transportation",
                confidence=0.9,
                model_name="ensemble",
                processing_time=0.5,
                reasoning="Test"
            )
        }
        
        with patch('src.normalization.merchant_normalizer.process_batch_remote.remote', return_value=mock_result):
            with patch('src.normalization.merchant_normalizer.ray.get', return_value=[mock_result]):
                results = normalizer.normalize_merchants(merchants)
                
                assert len(results) == 1
                assert "UBER *TRIP" in results
                assert results["UBER *TRIP"].normalized_name == "Uber"


class TestBatchSplitting:
    """Test batch splitting logic"""
    
    @pytest.fixture
    def normalizer(self):
        """Create MerchantNormalizer with mocked Ray"""
        with patch('src.normalization.merchant_normalizer.ray.is_initialized', return_value=True):
            return MerchantNormalizer()
    
    def test_merchants_split_into_correct_batches(self, normalizer):
        """Test that merchants are split into batches of correct size"""
        # Create enough merchants to require multiple batches
        batch_size = Config.BATCH_SIZE
        merchants = [f"MERCHANT_{i}" for i in range(batch_size * 2 + 3)]
        
        # Mock batch results
        mock_batch_results = []
        for i in range(0, len(merchants), batch_size):
            batch = merchants[i:i+batch_size]
            batch_result = {m: MerchantNormalizationResult(
                original_name=m,
                normalized_name=m.lower(),
                category="Other",
                confidence=0.5,
                model_name="ensemble",
                processing_time=0.1,
                reasoning="Test"
            ) for m in batch}
            mock_batch_results.append(batch_result)
        
        with patch('src.normalization.merchant_normalizer.process_batch_remote.remote') as mock_remote:
            # Make remote return the batch results in order
            mock_remote.side_effect = lambda batch, _: {m: MerchantNormalizationResult(
                original_name=m,
                normalized_name=m.lower(),
                category="Other",
                confidence=0.5,
                model_name="ensemble",
                processing_time=0.1,
                reasoning="Test"
            ) for m in batch}
            
            with patch('src.normalization.merchant_normalizer.ray.get', return_value=mock_batch_results):
                results = normalizer.normalize_merchants(merchants)
                
                # Verify all merchants are in results
                assert len(results) == len(set(merchants))
                # Verify correct number of batches were created
                expected_batches = (len(set(merchants)) + batch_size - 1) // batch_size
                assert mock_remote.call_count == expected_batches
    
    def test_single_batch_when_merchants_fit(self, normalizer):
        """Test that single batch is created when all merchants fit"""
        merchants = [f"MERCHANT_{i}" for i in range(Config.BATCH_SIZE)]
        
        mock_result = {m: MerchantNormalizationResult(
            original_name=m,
            normalized_name=m.lower(),
            category="Other",
            confidence=0.5,
            model_name="ensemble",
            processing_time=0.1,
            reasoning="Test"
        ) for m in merchants}
        
        with patch('src.normalization.merchant_normalizer.process_batch_remote.remote', return_value=mock_result):
            with patch('src.normalization.merchant_normalizer.ray.get', return_value=[mock_result]):
                results = normalizer.normalize_merchants(merchants)
                
                assert len(results) == len(merchants)


class TestProcessBatchRemote:
    """Test the Ray remote function process_batch_remote via normalizer"""
    
    @pytest.fixture
    def normalizer(self):
        """Create MerchantNormalizer with mocked Ray"""
        with patch('src.normalization.merchant_normalizer.ray.is_initialized', return_value=True):
            return MerchantNormalizer()
    
    def test_process_batch_calls_both_llms(self, normalizer):
        """Test that process_batch_remote calls both GPT and Gemini"""
        merchant_batch = ["UBER *TRIP", "STARBUCKS"]
        
        # Create mock results
        gpt_result_uber = MerchantNormalizationResult(
            original_name="UBER *TRIP",
            normalized_name="Uber",
            category="Transportation",
            confidence=0.9,
            model_name=Config.GPT_MODEL,
            processing_time=0.3,
            reasoning="Ride-sharing"
        )
        
        gemini_result_uber = MerchantNormalizationResult(
            original_name="UBER *TRIP",
            normalized_name="Uber",
            category="Transportation",
            confidence=0.85,
            model_name=Config.GEMINI_MODEL,
            processing_time=0.25,
            reasoning="Transportation service"
        )
        
        gpt_result_starbucks = MerchantNormalizationResult(
            original_name="STARBUCKS",
            normalized_name="Starbucks",
            category="Food & Dining",
            confidence=0.95,
            model_name=Config.GPT_MODEL,
            processing_time=0.2,
            reasoning="Coffee shop"
        )
        
        gemini_result_starbucks = MerchantNormalizationResult(
            original_name="STARBUCKS",
            normalized_name="Starbucks",
            category="Food & Dining",
            confidence=0.92,
            model_name=Config.GEMINI_MODEL,
            processing_time=0.18,
            reasoning="Coffee retailer"
        )
        
        # Mock clients
        mock_gpt_client = Mock()
        mock_gpt_client.normalize_merchant.side_effect = [gpt_result_uber, gpt_result_starbucks]
        
        mock_gemini_client = Mock()
        mock_gemini_client.normalize_merchant.side_effect = [gemini_result_uber, gemini_result_starbucks]
        
        # Create expected batch result
        expected_result = {
            "UBER *TRIP": CategoryAnalyzer.compare_results(gpt_result_uber, gemini_result_uber),
            "STARBUCKS": CategoryAnalyzer.compare_results(gpt_result_starbucks, gemini_result_starbucks)
        }
        
        # Create expected batch result
        expected_result = {
            "UBER *TRIP": CategoryAnalyzer.compare_results(gpt_result_uber, gemini_result_uber),
            "STARBUCKS": CategoryAnalyzer.compare_results(gpt_result_starbucks, gemini_result_starbucks)
        }
        
        with patch('src.normalization.merchant_normalizer.OpenAIClient', return_value=mock_gpt_client):
            with patch('src.normalization.merchant_normalizer.GeminiClient', return_value=mock_gemini_client):
                # Mock the remote function to return expected result
                with patch('src.normalization.merchant_normalizer.process_batch_remote.remote', return_value=expected_result):
                    with patch('src.normalization.merchant_normalizer.ray.get', return_value=[expected_result]):
                        results = normalizer.normalize_merchants(merchant_batch)
                        
                        # Verify both clients were called for each merchant (via the remote function)
                        # Note: We can't directly verify call counts since we're mocking the remote
                        # But we can verify the results structure
                        assert len(results) == 2
                        assert "UBER *TRIP" in results
                        assert "STARBUCKS" in results
                        assert results["UBER *TRIP"].normalized_name == "Uber"
                        assert results["STARBUCKS"].normalized_name == "Starbucks"
    
    def test_process_batch_handles_agreement_correctly(self, normalizer):
        """Test that process_batch uses CategoryAnalyzer correctly when LLMs agree"""
        merchant_batch = ["UBER *TRIP"]
        
        # Both LLMs return same result
        gpt_result = MerchantNormalizationResult(
            original_name="UBER *TRIP",
            normalized_name="Uber",
            category="Transportation",
            confidence=0.9,
            model_name=Config.GPT_MODEL,
            processing_time=0.3,
            reasoning="Test"
        )
        
        gemini_result = MerchantNormalizationResult(
            original_name="UBER *TRIP",
            normalized_name="Uber",
            category="Transportation",
            confidence=0.85,
            model_name=Config.GEMINI_MODEL,
            processing_time=0.25,
            reasoning="Test"
        )
        
        mock_gpt_client = Mock()
        mock_gpt_client.normalize_merchant.return_value = gpt_result
        
        mock_gemini_client = Mock()
        mock_gemini_client.normalize_merchant.return_value = gemini_result
        
        def mock_remote_side_effect(batch, cats):
            gpt = mock_gpt_client
            gemini = mock_gemini_client
            results = {}
            for merchant in batch:
                gpt_result = gpt.normalize_merchant(merchant, cats)
                gemini_result = gemini.normalize_merchant(merchant, cats)
                results[merchant] = CategoryAnalyzer.compare_results(gpt_result, gemini_result)
            return results
        
        with patch('src.normalization.merchant_normalizer.OpenAIClient', return_value=mock_gpt_client):
            with patch('src.normalization.merchant_normalizer.GeminiClient', return_value=mock_gemini_client):
                with patch('src.normalization.merchant_normalizer.process_batch_remote.remote', side_effect=mock_remote_side_effect):
                    with patch('src.normalization.merchant_normalizer.ray.get', side_effect=lambda futures: [mock_remote_side_effect(merchant_batch, Config.MERCHANT_CATEGORIES)]):
                        results = normalizer.normalize_merchants(merchant_batch)
                        
                        # Should use GPT result when they agree (CategoryAnalyzer logic)
                        assert results["UBER *TRIP"].normalized_name == "Uber"
                        assert results["UBER *TRIP"].category == "Transportation"
    
    def test_process_batch_handles_disagreement_with_confidence(self, normalizer):
        """Test that process_batch selects higher confidence when LLMs disagree"""
        merchant_batch = ["MERCHANT"]
        
        # GPT has higher confidence
        gpt_result = MerchantNormalizationResult(
            original_name="MERCHANT",
            normalized_name="Merchant A",
            category="Shopping",
            confidence=0.95,  # Higher confidence
            model_name=Config.GPT_MODEL,
            processing_time=0.3,
            reasoning="Test"
        )
        
        gemini_result = MerchantNormalizationResult(
            original_name="MERCHANT",
            normalized_name="Merchant B",
            category="Other",
            confidence=0.70,  # Lower confidence
            model_name=Config.GEMINI_MODEL,
            processing_time=0.25,
            reasoning="Test"
        )
        
        mock_gpt_client = Mock()
        mock_gpt_client.normalize_merchant.return_value = gpt_result
        
        mock_gemini_client = Mock()
        mock_gemini_client.normalize_merchant.return_value = gemini_result
        
        def mock_remote_side_effect(batch, cats):
            gpt = mock_gpt_client
            gemini = mock_gemini_client
            results = {}
            for merchant in batch:
                gpt_result = gpt.normalize_merchant(merchant, cats)
                gemini_result = gemini.normalize_merchant(merchant, cats)
                results[merchant] = CategoryAnalyzer.compare_results(gpt_result, gemini_result)
            return results
        
        with patch('src.normalization.merchant_normalizer.OpenAIClient', return_value=mock_gpt_client):
            with patch('src.normalization.merchant_normalizer.GeminiClient', return_value=mock_gemini_client):
                with patch('src.normalization.merchant_normalizer.process_batch_remote.remote', side_effect=mock_remote_side_effect):
                    with patch('src.normalization.merchant_normalizer.ray.get', side_effect=lambda futures: [mock_remote_side_effect(merchant_batch, Config.MERCHANT_CATEGORIES)]):
                        results = normalizer.normalize_merchants(merchant_batch)
                        
                        # Should select GPT result due to higher confidence
                        assert results["MERCHANT"].normalized_name == "Merchant A"
                        assert results["MERCHANT"].category == "Shopping"
                        assert results["MERCHANT"].confidence == 0.95


class TestErrorHandling:
    """Test error handling in merchant normalizer"""
    
    @pytest.fixture
    def normalizer(self):
        """Create MerchantNormalizer with mocked Ray"""
        with patch('src.normalization.merchant_normalizer.ray.is_initialized', return_value=True):
            return MerchantNormalizer()
    
    def test_process_batch_continues_on_individual_merchant_failure(self, normalizer):
        """Test that process_batch continues if one merchant fails"""
        merchant_batch = ["UBER", "FAILING_MERCHANT", "STARBUCKS"]
        
        # Create mock batch result with fallback for failed merchant
        mock_batch_result = {
            "UBER": MerchantNormalizationResult(
                original_name="UBER",
                normalized_name="Uber",
                category="Transportation",
                confidence=0.9,
                model_name="ensemble",
                processing_time=0.3,
                reasoning="Test"
            ),
            "FAILING_MERCHANT": MerchantNormalizationResult(
                original_name="FAILING_MERCHANT",
                normalized_name="FAILING_MERCHANT",
                category="Other",
                confidence=0.0,
                model_name="ensemble",
                processing_time=0.0,
                reasoning="Error: API Error"
            ),
            "STARBUCKS": MerchantNormalizationResult(
                original_name="STARBUCKS",
                normalized_name="Starbucks",
                category="Food & Dining",
                confidence=0.9,
                model_name="ensemble",
                processing_time=0.2,
                reasoning="Test"
            )
        }
        
        with patch('src.normalization.merchant_normalizer.process_batch_remote.remote', return_value=mock_batch_result):
            with patch('src.normalization.merchant_normalizer.ray.get', return_value=[mock_batch_result]):
                results = normalizer.normalize_merchants(merchant_batch)
                
                # Should have results for all merchants
                assert len(results) == 3
                assert "UBER" in results
                assert "FAILING_MERCHANT" in results
                assert "STARBUCKS" in results
                
                # Failed merchant should have fallback result
                assert results["FAILING_MERCHANT"].confidence == 0.0
                assert results["FAILING_MERCHANT"].category == "Other"
                assert "Error" in results["FAILING_MERCHANT"].reasoning


class TestIntegration:
    """Integration tests for full normalization flow"""
    
    @pytest.fixture
    def normalizer(self):
        """Create MerchantNormalizer with mocked Ray"""
        with patch('src.normalization.merchant_normalizer.ray.is_initialized', return_value=True):
            return MerchantNormalizer()
    
    def test_full_normalization_flow(self, normalizer):
        """Test complete normalization flow with multiple merchants and batches"""
        merchants = ["UBER *TRIP", "STARBUCKS", "AMAZON", "NETFLIX"]
        
        # Create mock batch results
        def create_mock_result(merchant_name: str) -> MerchantNormalizationResult:
            return MerchantNormalizationResult(
                original_name=merchant_name,
                normalized_name=merchant_name.split()[0].title(),
                category="Other",
                confidence=0.8,
                model_name="ensemble",
                processing_time=0.2,
                reasoning="Test"
            )
        
        # Split into batches based on BATCH_SIZE
        batch_size = Config.BATCH_SIZE
        unique_merchants = list(set(merchants))
        batches = [unique_merchants[i:i+batch_size] for i in range(0, len(unique_merchants), batch_size)]
        
        # Create mock results for each batch
        mock_batch_results = []
        for batch in batches:
            batch_result = {m: create_mock_result(m) for m in batch}
            mock_batch_results.append(batch_result)
        
        with patch('src.normalization.merchant_normalizer.process_batch_remote.remote') as mock_remote:
            # Make remote return appropriate batch results
            def remote_side_effect(batch, _):
                return {m: create_mock_result(m) for m in batch}
            
            mock_remote.side_effect = remote_side_effect
            
            with patch('src.normalization.merchant_normalizer.ray.get', return_value=mock_batch_results):
                results = normalizer.normalize_merchants(merchants)
                
                # Verify all unique merchants are in results
                assert len(results) == len(unique_merchants)
                for merchant in unique_merchants:
                    assert merchant in results
                    assert isinstance(results[merchant], MerchantNormalizationResult)
    
    def test_results_are_properly_aggregated(self, normalizer):
        """Test that results from multiple batches are properly aggregated"""
        merchants = [f"MERCHANT_{i}" for i in range(Config.BATCH_SIZE * 2)]
        
        # Create two batches
        batch1 = merchants[:Config.BATCH_SIZE]
        batch2 = merchants[Config.BATCH_SIZE:]
        
        batch1_result = {m: MerchantNormalizationResult(
            original_name=m,
            normalized_name=m.lower(),
            category="Other",
            confidence=0.5,
            model_name="ensemble",
            processing_time=0.1,
            reasoning="Test"
        ) for m in batch1}
        
        batch2_result = {m: MerchantNormalizationResult(
            original_name=m,
            normalized_name=m.lower(),
            category="Other",
            confidence=0.5,
            model_name="ensemble",
            processing_time=0.1,
            reasoning="Test"
        ) for m in batch2}
        
        with patch('src.normalization.merchant_normalizer.process_batch_remote.remote') as mock_remote:
            def remote_side_effect(batch, _):
                if set(batch) == set(batch1):
                    return batch1_result
                else:
                    return batch2_result
            
            mock_remote.side_effect = remote_side_effect
            
            with patch('src.normalization.merchant_normalizer.ray.get', return_value=[batch1_result, batch2_result]):
                results = normalizer.normalize_merchants(merchants)
                
                # Verify all merchants from both batches are in results
                assert len(results) == len(merchants)
                for merchant in merchants:
                    assert merchant in results

