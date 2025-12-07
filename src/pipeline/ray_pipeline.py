"""
Main data processing pipeline using Ray.

This is the main code that ingests csv, validates data, normalizes it in parallel, and 
returns a clean output that can be stored in an S3 storage.

1. Ingest messy CSV
2. Validate each row with security first principles and ensuring data can be processed by the system
3. Normalize dates and amounts using helper functions inside of date_normalizer and amount_normalizer respectively
4. Apply distributed computing for merchant normalization using Ray + dual LLMs in merchant_normalizer 
5. Return clean DataFrame and prepare it for S3 storage.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import time

from src.config import Config
from src.validation.security_validator import get_security_validator
from src.validation.data_validator import get_data_validator
from src.normalization.date_normalizer import DateNormalizer
from src.normalization.amount_normalizer import AmountNormalizer
from src.normalization.merchant_normalizer import MerchantNormalizer
from src.llm.base_llm import MerchantNormalizationResult

logger = logging.getLogger(__name__)


class CentralPipeline:
    """
    Main pipeline for processing messy financial data.
    
    Pipeline stages re-iterated:
    1. Load CSV
    2. Security validation (injection prevention, traversal prevention, prompt injection prevention)
    3. Validate data and note issues
    4. Normalize dates with fast date parsing utilities and helper functions
    5. Normalize amounts with helper functions and regex
    6. Normalize merchants using Ray + dual LLMs
    7. Return clean DataFrame
    """
    
    def __init__(self):
        """Initialize pipeline with all components"""
        self.security_validator = get_security_validator()
        self.data_validator = get_data_validator()
        self.date_normalizer = DateNormalizer()
        self.amount_normalizer = AmountNormalizer()
        self.merchant_normalizer = MerchantNormalizer()
        
        logger.info("Pipeline initialized")
    
    def process_csv(self, csv_path: Path) -> Tuple[pd.DataFrame, Dict]:
        """
        Process a CSV file through the complete pipeline.
        
        Args:
            csv_path: Path to messy CSV file
            
        Returns:
            Tuple of (clean_dataframe, metadata_dict)
            - clean_dataframe: Normalized transactions
            - metadata_dict: Processing stats, errors, timing (we get these from the data validation step)
        """
        start_time = time.time()
        metadata = {
            'input_file': str(csv_path),
            'total_rows': 0,
            'validation_errors': [],
            'processing_time_seconds': 0.0
        }
        
        logger.info(f"Processing CSV: {csv_path}")
        
        # check the input path
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
        
        # enforce correct columns in data
        required_columns = ['date', 'merchant', 'amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Preserve original row index for ground truth matching
        df['original_row_index'] = df.index
        
        # Preserve original row index for ground truth matching
        df['original_row_index'] = df.index
        
        metadata['total_rows'] = len(df)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Security validation
        logger.info("Starting Security validation")
        valid_merchants, invalid_entries = self.security_validator.validate_batch(df['merchant'].tolist())
        metadata['validation_errors'].extend([f"Security issue at row {idx}: {merchant} - {', '.join(issues)}" 
                                               for idx, merchant, issues in invalid_entries])
        logger.info(f"Security validation complete: {len(invalid_entries)} issues found")
        
        # Remove dangerous merchants
        dangerous_merchants = {merchant for _, merchant, _ in invalid_entries}
        df = df[~df['merchant'].isin(dangerous_merchants)].copy()
        logger.info(f"Removed {len(dangerous_merchants)} dangerous merchants")
        
        # data validation
        # Fill NaN values before converting to dict to avoid issues
        df_for_validation = df.fillna("")
        validation_summary = self.data_validator.validate_batch(df_for_validation.to_dict('records'))
        metadata['validation_errors'].extend([f"Data quality issue at row {idx}" 
                                               for idx in validation_summary.get('critical_indices', [])])
        logger.info(f"Data quality validation complete: {validation_summary.get('with_issues', 0)} rows with issues")
        
        # date normalization
        logger.info("Starting Dates Normalization")
        df['normalized_date'] = df['date'].apply(self.date_normalizer.normalize_date)
        logger.info("Dates normalization complete")

        # amounts normalization
        logger.info("Starting amount normalization")
        # amount_normalizer.normalize() returns (amount, currency) tuple
        amount_results = df['amount'].apply(self.amount_normalizer.normalize)
        df['normalized_amount'] = amount_results.apply(lambda x: x[0] if isinstance(x, tuple) else x)
        logger.info("Amount normalization complete")
        
        # merchant normalization
        logger.info("Starting merchant normalization")
        merchant_list = df['merchant'].tolist()
        merchant_results = self.merchant_normalizer.normalize_merchants(merchant_list)
        logger.info("Merchant normalization complete")
        
        # Map results back to DataFrame
        # merchant_results is a Dict[str, MerchantNormalizationResult]
        # Use pandas .map() with dict.get() for efficient lookup with fallback
        # This is cleaner than a separate function - pandas handles the mapping efficiently
        metadata['unique_merchants'] = len(merchant_results)
        metadata['total_amount'] = df['normalized_amount'].sum()

        def get_result(merchant: str) -> MerchantNormalizationResult:
            """Get normalization result with fallback - used by pandas map()"""
            return merchant_results.get(
                merchant,
                MerchantNormalizationResult(
                    original_name=merchant,
                    normalized_name=merchant,
                    category="Other",
                    confidence=0.0,
                    model_name="fallback",
                    processing_time=0.0,
                    reasoning="Merchant not found in results"
                )
            )
        
        # Map once to get all results, then extract fields (more efficient than 3 separate maps)
        results_series = df['merchant'].map(get_result)
        df['normalized_merchant'] = results_series.map(lambda r: r.normalized_name)
        df['category'] = results_series.map(lambda r: r.category)
        df['confidence'] = results_series.map(lambda r: r.confidence)
        
        # create clean output DataFrame (preserve original_row_index for ground truth matching)
        clean_df = df[[
            'normalized_date',
            'normalized_merchant', 
            'normalized_amount',
            'category',
            'original_row_index'
        ]].copy()

        # Rename columns for clarity
        clean_df.columns = ['date', 'merchant', 'amount', 'category', 'original_row_index']
        
        # important metrics for the final analysis!
        top_category = clean_df.groupby('category')['amount'].sum().idxmax()
        metadata['top_spending_category'] = top_category
        metadata['top_category_amount'] = clean_df[clean_df['category'] == top_category]['amount'].sum()
        
        # Calculate processing time
        metadata['processing_time_seconds'] = time.time() - start_time
        
        logger.info(f"Pipeline complete in {metadata['processing_time_seconds']:.2f}s")
        
        return clean_df, metadata
    
    def shutdown(self, shutdown_ray: bool = False):
        """
        Shutdown pipeline components.
        
        Args:
            shutdown_ray: If True, shutdown Ray. If False, keep Ray alive for reuse.
        """
        if shutdown_ray:
            self.merchant_normalizer.shutdown()
            logger.info("Pipeline shutdown complete (Ray shutdown)")


# Helper function for CLI
def process_financial_data(csv_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Helper function to process a CSV file.
    
    Args:
        csv_path: Path to CSV file (string)
        
    Returns:
        Tuple of (clean_dataframe, metadata)
    """
    pipeline = CentralPipeline()
    
    try:
        df, metadata = pipeline.process_csv(Path(csv_path))
        return df, metadata
    finally:
        # Don't shutdown Ray - keep it alive for reuse
        pipeline.shutdown(shutdown_ray=False)


def shutdown_ray():
    """Shutdown Ray cluster. Call this when exiting the CLI."""
    import ray
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray cluster shutdown complete")