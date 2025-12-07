"""
Ensemble voting algorithm for merchant normalization.

This module compares results from GPT and Gemini LLMs and selects the best result
using ensemble voting logic. Also includes metrics to evaluate AI accuracy.
"""

import logging
from typing import Tuple
from src.llm.base_llm import MerchantNormalizationResult

logger = logging.getLogger(__name__)


class CategoryAnalyzer:
    """
    Ensemble voting system for comparing LLM results.
    
    Not an LLM client - this is a utility class for decision-making.
    """
    
    @staticmethod
    def compare_results(
        gpt_result: MerchantNormalizationResult,
        gemini_result: MerchantNormalizationResult
    ) -> MerchantNormalizationResult:
        """
        Compare GPT and Gemini results using ensemble voting.

        Args:
            gpt_result: Result from gpt-5-nano
            gemini_result: Result from gemini-2.5-flash-lite
            
        Returns:
            Selected MerchantNormalizationResult (the "winning" result)
        """
        # Check if both category AND normalized_name agree
        categories_agree = gpt_result.category == gemini_result.category
        names_agree = gpt_result.normalized_name == gemini_result.normalized_name
        
        if categories_agree and names_agree:
            # both are the same, just return one result
            return gpt_result
        else:
            # confidence-based selection
            if gpt_result.confidence > gemini_result.confidence:
                logger.warning(
                    f"Disagreement: GPT=[category='{gpt_result.category}', name='{gpt_result.normalized_name}', conf={gpt_result.confidence}] vs "
                    f"Gemini=[category='{gemini_result.category}', name='{gemini_result.normalized_name}', conf={gemini_result.confidence}]. "
                    f"Selecting GPT due to higher confidence."
                )
                return gpt_result
            else:
                # Gemini has higher or equal confidence
                logger.warning(
                    f"Disagreement: GPT=[category='{gpt_result.category}', name='{gpt_result.normalized_name}', conf={gpt_result.confidence}] vs "
                    f"Gemini=[category='{gemini_result.category}', name='{gemini_result.normalized_name}', conf={gemini_result.confidence}]. "
                    f"Selecting Gemini due to higher/equal confidence."
                )
                return gemini_result
    
    @staticmethod
    def calculate_agreement_rate(
        results_pairs: list[Tuple[MerchantNormalizationResult, MerchantNormalizationResult]]
    ) -> float:
        """
        Helper function for AI evaluation metrics. We calculate agreement rate between 
        GPT and Gemini across multiple results.
        
        Args:
            results_pairs: List of (gpt_result, gemini_result) tuples
            
        Returns:
            Agreement rate as float between 0.0 and 1.0
        """
        if not results_pairs:
            return 0.0
        
        agreements = sum(
            1 for gpt, gemini in results_pairs
            if gpt.category == gemini.category and gpt.normalized_name == gemini.normalized_name
        )
        
        agreement_rate = agreements / len(results_pairs)
        logger.info(f"Agreement rate: {agreements}/{len(results_pairs)} = {agreement_rate:.2%}")
        
        return agreement_rate