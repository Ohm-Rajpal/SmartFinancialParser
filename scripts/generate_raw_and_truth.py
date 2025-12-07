#!/usr/bin/env python3
"""
Generate messy financial transaction data for testing.

This script creates a CSV file with intentionally inconsistent formatting:
- 10+ different date formats
- Mixed currency symbols and spacing
- Merchant name variations and typos
- Edge cases (empty values, injection attempts, special characters)

Spec Requirements:
- Dates: Mix formats (e.g., 2023-01-01, Jan 1st 23, 01/01/2023)
- Merchants: Mix naming conventions (e.g., UBER *TRIP, Uber Technologies, UBER EATS)
- Amounts: Include currency symbols and inconsistent spacing

Usage:
    python scripts/generate_messy_data.py
    
Output:
    data/raw/messy_transactions.csv (3 columns: date, merchant, amount)
"""

import random
import csv
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import Config


class MessyDataGenerator:
    """Generates intentionally messy transaction data for testing"""
    
    # ground truth for AI evaluation
    # required to be hardcoded to truly benchmark AI
    
    # Security test casess
    EDGE_CASE_MERCHANTS = [
        "=1+1",                # CSV injection
        "=cmd|'/c calc'!A1",   # CSV formula injection
        "@SUM(A1:A10)",        # CSV function injection
        "Normal Store & Co.",  # Ampersand
        "Store-With-Dashes",
        "Store's Apostrophe",
        "Ürban Çafé",   # Unicode characters
        "",             # Empty merchant (missing data)
        "   ",          # Whitespace only
        "A" * 250,      # Very long name (exceeds max length)
    ]
    
    def __init__(self, num_transactions):
        """
        Initialize generator
        Args:
            num_transactions: Number of transactions to generate
        """
        self.num_transactions = num_transactions if num_transactions else Config.NUM_TEST_TRANSACTIONS
        self.output_path = Config.RAW_DATA_DIR / "messy_transactions.csv"
        
    def generate_random_date(self) -> str:
        """Generate a random date in the past year with random format"""
        # Random date within last year
        days_ago = random.randint(0, 365)
        date = datetime.now() - timedelta(days=days_ago)
        
        # Pick random format
        date_format = random.choice(Config.DATE_FORMATS)
        
        # Add some extra messiness
        date_str = date.strftime(date_format)
        
        # Add extra spaces or strange formatting
        if random.random() < 0.1:  # 10% chance
            messiness = random.choice([
                f" {date_str} ",  # Extra spaces
                f"{date_str}  ",  # Trailing spaces
                f"  {date_str}",  # Leading spaces
                date_str.replace("-", " - "),  # Spaces around delimiters
            ])
            return messiness
        
        return date_str
    
    def generate_random_amount(self) -> str:
        """Generate a random amount with random currency formatting"""
        # Generate amount (with some negative refunds)
        if random.random() < 0.05:  # 5% refunds
            amount = -random.uniform(5, 500)
            is_negative = True
        else:
            amount = random.uniform(1, 500)
            is_negative = False
        
        # Work with absolute value for formatting
        abs_amount = abs(amount)
        
        # Pick random currency symbol
        currency = random.choice(Config.CURRENCY_SYMBOLS)
        
        # Format with various styles
        style = random.choice([
            "symbol_before",  # $123.45 or -$123.45
            "symbol_after",   # 123.45 USD or -123.45 USD
            "no_symbol",      # 123.45 or -123.45
            "weird_spacing",  # $ 123.45 or -$ 123.45
            "european",       # 123,45 or -123,45
            "negative_after"  # $123.45- (wrong but realistic typo)
        ])
        
        if style == "symbol_before":
            # Standard: $123.45 or -$123.45
            formatted = f"{currency}{abs_amount:,.2f}"
            if is_negative:
                formatted = f"-{formatted}"
                
        elif style == "symbol_after":
            # After: 123.45 USD or -123.45 USD
            formatted = f"{abs_amount:,.2f} {currency}"
            if is_negative:
                formatted = f"-{formatted}"
                
        elif style == "no_symbol":
            # No symbol: 123.45 or -123.45
            formatted = f"{abs_amount:,.2f}"
            if is_negative:
                formatted = f"-{formatted}"
                
        elif style == "weird_spacing":
            # Weird spacing: $ 123.45 or - $ 123.45
            spaces = random.choice([" ", "  ", "   "])
            formatted = f"{currency}{spaces}{abs_amount:,.2f}"
            if is_negative:
                # Negative with weird spacing: - $ 123.45 or -$ 123.45
                if random.random() < 0.5:
                    formatted = f"-{spaces}{formatted}"
                else:
                    formatted = f"-{formatted}"
                    
        elif style == "negative_after":
            # Wrong placement: $123.45- (common data entry error)
            formatted = f"{currency}{abs_amount:,.2f}"
            if is_negative:
                formatted = f"{formatted}-"
            # If positive, just use standard format
            else:
                formatted = f"{currency}{abs_amount:,.2f}"
                
        else:  # european
            # European format: 123,45 instead of 123.45
            formatted = f"{currency}{abs_amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            if is_negative:
                formatted = f"-{formatted}"
        
        # Occasionally add extra spaces
        if random.random() < 0.1:
            formatted = f" {formatted} "
        
        return formatted
    
    def generate_random_merchant(self) -> str:
        """Generate a random merchant name with variations"""
        # 90% real merchants, 10% edge cases
        if random.random() < 0.9:
            # Pick a random merchant
            merchant_data = random.choice(Config.MERCHANTS_WITH_CATEGORIES)
            # Pick a random variation
            merchant = random.choice(merchant_data["variations"])
        else:
            # Edge case merchant
            merchant = random.choice(self.EDGE_CASE_MERCHANTS)
        
        # Occasionally add extra whitespace
        if random.random() < 0.05:
            merchant = f"  {merchant}  "
        
        return merchant
    

    def generate_transaction(self) -> tuple:
        """
        Generate transaction with explicit category mapping.
        
        Ground truth uses hardcoded categories to benchmark the AI.
        """
        # Pick random merchant
        merchant_data = random.choice(Config.MERCHANTS_WITH_CATEGORIES)
        
        # Pick random variation
        messy_name = random.choice(merchant_data["variations"])
        
        # Ground truth from explicit category mapping
        ground_truth = {
            "clean_merchant": merchant_data["clean_name"],
            "true_category": merchant_data["category"]
        }
        
        transaction = {
            "date": self.generate_random_date(),
            "merchant": messy_name,
            "amount": self.generate_random_amount()
        }
        
        return transaction, ground_truth
    
    def generate_csv(self) -> tuple:
        """
        Generate both messy CSV and ground truth CSV. This is crucial to evaluate
        performance of my AI powered data pipeline.
        
        Returns:
            Tuple of (messy_csv_path, ground_truth_csv_path)
        """
        print(f"Generating {self.num_transactions} messy transactions...")
        
        transactions = []
        ground_truths = []
        
        for i in range(self.num_transactions):
            txn, gt = self.generate_transaction()
            
            # Add row index to ground truth
            gt["row_index"] = i
            gt["messy_merchant"] = txn["merchant"]
            
            transactions.append(txn)
            ground_truths.append(gt)
        
        # Save messy CSV
        messy_df = pd.DataFrame(transactions)
        messy_df.to_csv(self.output_path, index=False)
        
        print(f"   Generated messy data: {self.output_path}")
        print(f"   Total transactions: {self.num_transactions}")
        print(f"   File size: {self.output_path.stat().st_size:,} bytes")
        
        # Save ground truth CSV
        gt_path = Config.GROUND_TRUTH_DIR / "ground_truth.csv"
        gt_df = pd.DataFrame(ground_truths)
        gt_df.to_csv(gt_path, index=False)
        
        print(f"   Generated ground truth: {gt_path}")
        print(f"   Ground truth rows: {len(ground_truths)}")
        
        self._print_statistics(transactions)
        self._print_ground_truth_statistics(ground_truths)
        
        return self.output_path, gt_path
    
    def _print_statistics(self, transactions: list):
        """Print statistics about generated data"""
        print("\nData Statistics:")
        
        # Count unique date formats
        date_formats_used = set()
        for txn in transactions:
            # Try to detect which format was used (rough estimate)
            date_str = txn["date"].strip()
            if "/" in date_str:
                date_formats_used.add("slash format")
            elif "-" in date_str:
                date_formats_used.add("dash format")
            elif "." in date_str:
                date_formats_used.add("dot format")
        
        print(f"   Date format variations: {len(date_formats_used)}+")
        
        # Count unique merchants
        unique_merchants = len(set(txn["merchant"].strip() for txn in transactions))
        print(f"   Unique merchant names: {unique_merchants}")
        
        # Count empty merchants
        empty_merchants = sum(1 for txn in transactions if not txn["merchant"].strip())
        print(f"   Empty merchants: {empty_merchants}")
        
        # Count edge cases
        edge_cases = sum(1 for txn in transactions 
                        if any(txn["merchant"].startswith(ec) for ec in ["=", "@", "+", "-"]))
        print(f"   CSV injection attempts: {edge_cases}")
        
        # Count negative amounts
        negative_amounts = sum(1 for txn in transactions if "-" in txn["amount"])
        print(f"   Negative amounts (refunds): {negative_amounts}")
        
        # Count currency variations
        currency_symbols = set()
        for txn in transactions:
            for symbol in Config.CURRENCY_SYMBOLS:
                if symbol in txn["amount"]:
                    currency_symbols.add(symbol)
        print(f"   Currency symbols used: {len(currency_symbols)}")
        
        print("\nMessy data tests:")
        print("Date normalization (10+ formats)")
        print("Amount parsing (currency symbols, spacing, negatives)")
        print("Merchant normalization (variations, typos)")
        print("Missing data handling (empty merchants)")
        print("Security validation (CSV injection attempts)")
        print("Unicode handling (special characters)")
        print("Refund/negative amount handling")

    def _print_ground_truth_statistics(self, ground_truths: list):
        """Print statistics about ground truth labels"""
        print("\nGround Truth Statistics:")
        
        # Count unique clean merchants
        unique_clean = len(set(gt["clean_merchant"] for gt in ground_truths))
        print(f"   Unique clean merchants: {unique_clean}")
        
        
        category_counts = Counter(gt["true_category"] for gt in ground_truths)
        
        print(f"   Category distribution (from explicit mapping):")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(ground_truths)) * 100
            print(f"      {category:30s} {count:4d} ({percentage:5.1f}%)")

def main():
    """Main entry point"""
    print("=" * 70)
    print("Messy Financial Data Generator")
    print("=" * 70)
    print("\nGenerating CSV with 3 columns: date, merchant, amount")
    print("(No description column - as per spec requirements)\n")
    
    # Ensure directories exist
    Config.ensure_directories()
    
    # Generate data
    generator = MessyDataGenerator(num_transactions=500)
    messy_csv_path, ground_truth_path = generator.generate_csv()
    
    print("\n" + "=" * 70)
    print(f"  Success! Generated data:")
    print(f"   Messy CSV: {messy_csv_path}")
    print(f"   Ground Truth: {ground_truth_path}")
    print("\n   CSV Format:")
    print("   Columns: date, merchant, amount")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())