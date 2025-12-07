"""
Merchant normalization using Ray for distributed processing.

Using Ray to parallelize LLM calls to both
GPT and Gemini, then use ensemble voting to select optimal result.
"""

import logging
import time
from typing import List, Dict, Tuple
from collections import Counter
import ray

from src.config import Config
from src.llm.openai_client import OpenAIClient
from src.llm.gemini_client import GeminiClient
from src.llm.model_comparator import CategoryAnalyzer
from src.llm.base_llm import MerchantNormalizationResult

logger = logging.getLogger(__name__)


class MerchantNormalizer:
    """
    Normalizes merchant names using parallel LLM calls with Ray.
    
    High level architecture:
    1. Avoid double processing merchants
    2. Split batches for Ray parallelization
    3. Each Ray worker processes one batch:
       - For a merchant, call GPT and Gemini sequentially
       - Use logic from model_comparator for the ensemble voting algo
    4. Perform a reduce task by aggregating results from all Ray workers
    """
    
    def __init__(self):
        """Initialize merchant normalizer"""
        self.available_categories = Config.MERCHANT_CATEGORIES
        self.batch_size = Config.BATCH_SIZE

        # initialize ray
        if not ray.is_initialized():
            ray.init(num_cpus=Config.RAY_NUM_CPUS, 
                    object_store_memory=Config.RAY_OBJECT_STORE_MEMORY, 
                    logging_level=logging.INFO)    
        logger.info(f"Ray initialized with {Config.RAY_NUM_CPUS} CPUs")
    
    # core logic
    def normalize_merchants(
        self,
        merchant_names: List[str]
    ) -> Dict[str, MerchantNormalizationResult]:
        """
        Normalize a list of merchant names using Ray + dual LLMs.
        
        Args:
            merchant_names: List of raw merchant names (has duplicates potentially)
            
        Returns:
            Dict mapping original merchant name -> MerchantNormalizationResult
            
        Example:
            Input: ["UBER *TRIP", "UBER *TRIP", "STARBUCKS"]
            Output: {
                "UBER *TRIP": MerchantNormalizationResult(...),
                "STARBUCKS": MerchantNormalizationResult(...)
            }
        """
        start_time = time.time()
        
        # get unique merchants to avoid duplicate processing
        unique_merchants = set(merchant_names)
        unique_merchants_list = list(unique_merchants)
        logger.info(f"Processing {len(unique_merchants_list)} unique merchants from {len(merchant_names)} total")

        # split batches
        batches = [unique_merchants_list[i:i+Config.BATCH_SIZE] for i in range(0, len(unique_merchants_list), Config.BATCH_SIZE)]
        logger.info(f"Split into {len(batches)} batches of size: {Config.BATCH_SIZE}")
        
        # parallel processing
        futures = [process_batch_remote.remote(batch, self.available_categories) for batch in batches]

        # wait for all batches to complete
        batch_results = ray.get(futures)
        
        all_results = {}
    
        for batch_result in batch_results:
            all_results.update(batch_result)
        
        processing_time = time.time() - start_time
        logger.info(f"Normalized {len(all_results)} merchants in {processing_time:.2f}s")
        
        return all_results
    
    def shutdown(self):
        """Shutdown Ray"""
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")

@ray.remote
def process_batch_remote(
    merchant_batch: List[str],
    available_categories: List[str]
) -> Dict[str, MerchantNormalizationResult]:
    """
    Ray remote function to process one batch of merchants.
    
    This runs in a separate Ray worker process.
    For EACH merchant in the batch:
    1. Call GPT to normalize
    2. Call Gemini to normalize
    3. Use ensemble voting to pick best result
    
    Args:
        merchant_batch: List of unique merchant names for this batch
        available_categories: Valid categories
        
    Returns:
        Dict mapping merchant -> final MerchantNormalizationResult
    """
    logger.info(f"Worker processing batch of {len(merchant_batch)} merchants")
    
    gpt_client = OpenAIClient()
    gemini_client = GeminiClient()
    results = {}
    
    for merchant in merchant_batch:
        gpt_result = gpt_client.normalize_merchant(merchant, available_categories)
        gemini_result = gemini_client.normalize_merchant(merchant, available_categories)
        # ensemble algo from model comparator
        results[merchant] = CategoryAnalyzer.compare_results(gpt_result, gemini_result)
    
    logger.info(f"Worker completed batch: {len(results)} merchants normalized")
    return results