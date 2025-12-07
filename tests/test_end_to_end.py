"""
End-to-end integration test for the complete pipeline.

Tests the full flow from messy CSV to clean normalized output.
"""

import sys
import logging
from pathlib import Path

# Add project root to path so imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



# INFO level shows pipeline progress and Ray worker outputs
logging.basicConfig(
    level=logging.INFO,  # Shows INFO, WARNING, ERROR (hides DEBUG)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # Override any existing logging config
)

from src.pipeline.ray_pipeline import process_financial_data

# Test with real messy data
df, metadata = process_financial_data("data/raw/messy_transactions.csv")

print("\n" + "="*60)
print("PIPELINE TEST RESULTS")
print("="*60)
print(f"Processed: {metadata['total_rows']} rows")
print(f"Time: {metadata['processing_time_seconds']:.2f}s")
print(f"Validation errors: {len(metadata['validation_errors'])}")

print("\nFirst 10 rows:")
print(df.head(10))

print("\nSpending by category:")
print(df.groupby('category')['amount'].sum().sort_values(ascending=False))

print("\nTop spending category:")
top_cat = df.groupby('category')['amount'].sum().idxmax()
top_amt = df[df['category'] == top_cat]['amount'].sum()
print(f"{top_cat}: ${top_amt:,.2f}")

# Save output
df.to_csv("data/processed/clean_transactions.csv", index=False)
print("\nSaved to: data/processed/clean_transactions.csv")
