"""
Merchant normalization using Ray for distributed processing.

Using Ray to parallelize LLM calls to both
GPT and Gemini, then use ensemble voting to select optimal result.
"""

import logging
import asyncio
import time
from typing import List, Dict
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
        self.base_batch_size = Config.BATCH_SIZE  # Base batch size from config (used as fallback)
        self.min_batch_size = 5   # Minimum merchants per batch
        self.max_batch_size = 200  # Maximum merchants per batch

        # Ray should already be initialized by CLI, but check just in case
        if not ray.is_initialized():
            logger.warning("Ray not initialized! Initializing now (should have been done by CLI)")
            ray.init(
                num_cpus=Config.RAY_NUM_CPUS, 
                object_store_memory=Config.RAY_OBJECT_STORE_MEMORY, 
                logging_level=logging.INFO
            )

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
        optimal_batch_size = self._calculate_optimal_batch_size(len(unique_merchants_list))
        batches = [unique_merchants_list[i:i+optimal_batch_size] for i in range(0, len(unique_merchants_list), 
                optimal_batch_size)]
        logger.info(f"Split into {len(batches)} batches of size: {optimal_batch_size}")
        
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

    def _calculate_optimal_batch_size(self, total_merchants: int) -> int:
        """
        Helper function to calculate optimal batch size based on number of merchants
        which will improve parallelism.
        
        Args:
            total_merchants: Total number of unique merchants to process
            
        Returns:
            Optimal batch size
        """
        num_cpus = Config.RAY_NUM_CPUS
        
        ideal_batch_size = max(1, total_merchants // num_cpus)
        
        # restrict batch size
        batch_size = max(self.min_batch_size, min(ideal_batch_size, self.max_batch_size))
        
        logger.info(f"Calculated optimal batch size: {batch_size} for {total_merchants} merchants on {num_cpus} CPUs")
        
        return batch_size

# Previously was using sequential processing but it was too slow
# Async processing will speed up the process
async def process_batch_async(
    merchant_batch: List[str],
    available_categories: List[str]
) -> Dict[str, MerchantNormalizationResult]:
    """
    Process batch with async LLM calls for 2x speedup than before
    
    Args:
        merchant_batch: List of merchant names to process
        available_categories: List of available categories

    Returns:
        Dict mapping original merchant name -> MerchantNormalizationResult
    """
    logger.info(f"Worker processing batch of {len(merchant_batch)} merchants (async)")
    
    gpt_client = OpenAIClient()
    gemini_client = GeminiClient()
    results = {}
    
    for merchant in merchant_batch:
        try:
            gpt_task = asyncio.to_thread(
                gpt_client.normalize_merchant,
                merchant,
                available_categories
            )
            gemini_task = asyncio.to_thread(
                gemini_client.normalize_merchant,
                merchant,
                available_categories
            )
            
            gpt_result, gemini_result = await asyncio.gather(gpt_task, gemini_task)
            
            # use ensemble voting to select the best result
            results[merchant] = CategoryAnalyzer.compare_results(gpt_result, gemini_result)
            
        except Exception as e:  # handle errors gracefully
            logger.error(f"Failed to normalize merchant '{merchant}': {e}")
            results[merchant] = MerchantNormalizationResult(
                original_name=merchant,
                normalized_name=merchant,
                category="Other",
                confidence=0.0,
                model_name="ensemble",
                processing_time=0.0,
                reasoning=f"Error: {str(e)}"
            )
    
    logger.info(f"Worker completed batch: {len(results)} merchants normalized")
    return results

@ray.remote
def process_batch_remote(
    merchant_batch: List[str],
    available_categories: List[str]
) -> Dict[str, MerchantNormalizationResult]:
    """
    Ray remote function to process one batch of merchants using async processing
    
    Args:
        merchant_batch: List of unique merchant names for this batch
        available_categories: Valid categories
        
    Returns:
        Dict mapping merchant -> final MerchantNormalizationResult
    """
    return asyncio.run(process_batch_async(merchant_batch, available_categories))