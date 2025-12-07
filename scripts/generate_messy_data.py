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
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config


class MessyDataGenerator:
    """Generates intentionally messy transaction data for testing"""
    
    # merchant names with common variations
    MERCHANTS = {
        "Starbucks": [
            "STARBUCKS",
            "Starbucks Coffee",
            "STARBUCKS #1234",
            "Starbucks Corp",
            "starbucks",
            "SBUX"
        ],
        "Uber": [
            "UBER *TRIP",
            "Uber Technologies",
            "UBER EATS",
            "Uber",
            "uber *trip 12345",
            "UBER BV"
        ],
        "Amazon": [
            "AMAZON.COM",
            "Amazon Marketplace",
            "AMZN Mktp US",
            "Amazon Prime",
            "amazon.com*123",
            "AMZ*Amazon"
        ],
        "McDonald's": [
            "MCDONALDS",
            "McDonald's #123",
            "MCD*McDonalds",
            "McDonalds F12345",
            "mc donalds"
        ],
        "Shell Gas": [
            "SHELL OIL",
            "Shell 12345678",
            "SHELL GAS STATION",
            "Shell - Richmond",
            "shell gas"
        ],
        "Walgreens": [
            "WALGREENS #123",
            "Walgreens Store",
            "WAG*Walgreens",
            "WALGREENS PHARMACY"
        ],
        "Target": [
            "TARGET",
            "Target Store",
            "TGT*TARGET",
            "Target #1234",
            "target.com"
        ],
        "Whole Foods": [
            "WHOLE FOODS",
            "Whole Foods Market",
            "WFM*Whole Foods",
            "WHOLEFDS"
        ],
        "Netflix": [
            "NETFLIX.COM",
            "Netflix Subscription",
            "NETFLIX *STREAMING",
            "Netflix Inc"
        ],
        "AT&T": [
            "ATT*BILL PAYMENT",
            "AT&T Wireless",
            "AT&T MOBILITY",
            "ATandT"
        ],
        "CVS Pharmacy": [
            "CVS/PHARMACY",
            "CVS #1234",
            "CVS STORE",
            "CVS/pharmacy"
        ],
        "Chevron": [
            "CHEVRON",
            "Chevron Gas",
            "CHEVRON #123456",
            "chevron station"
        ],
        "Safeway": [
            "SAFEWAY",
            "Safeway Store",
            "SAFEWAY #1234",
            "Safeway Inc"
        ],
        "Lyft": [
            "LYFT *RIDE",
            "Lyft Inc",
            "lyft *ride 12345",
            "LYFT"
        ],
        "Spotify": [
            "SPOTIFY",
            "Spotify USA",
            "SPOTIFY *PREMIUM",
            "spotify.com"
        ],
        "Costco": [
            "COSTCO WHSE",
            "Costco Wholesale",
            "COSTCO #123",
            "costco"
        ],
        "Home Depot": [
            "HOME DEPOT",
            "The Home Depot",
            "HOMEDEPOT #1234",
            "HD*Home Depot"
        ],
        "Trader Joe's": [
            "TRADER JOES",
            "Trader Joe's",
            "TJS*Trader Joes",
            "TRADERJOES"
        ],
        "Chipotle": [
            "CHIPOTLE",
            "Chipotle Mexican Grill",
            "CHIPOTLE #1234",
            "chipotle mexican"
        ],
        "Apple": [
            "APPLE.COM/BILL",
            "Apple Store",
            "APL*APPLE",
            "Apple Inc"
        ],
        "PG&E": [
            "PGE BILL PAYMENT",
            "PG&E Energy",
            "PACIFIC GAS ELECTRIC",
            "PG AND E"
        ]
    }
    
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
            # Pick a random base merchant
            base_merchant = random.choice(list(self.MERCHANTS.keys()))
            # Pick a random variation
            merchant = random.choice(self.MERCHANTS[base_merchant])
        else:
            # Edge case merchant
            merchant = random.choice(self.EDGE_CASE_MERCHANTS)
        
        # Occasionally add extra whitespace
        if random.random() < 0.05:
            merchant = f"  {merchant}  "
        
        return merchant
    
    def generate_transaction(self) -> dict:
        """Generate a single transaction with messy data (3 columns only)"""
        return {
            "date": self.generate_random_date(),
            "merchant": self.generate_random_merchant(),
            "amount": self.generate_random_amount()
        }
    
    def generate_csv(self) -> Path:
        """
        Generate messy CSV file with 3 columns: date, merchant, amount.
        
        Returns:
            Path to generated CSV file
        """
        print(f"Generating {self.num_transactions} messy transactions...")
        
        # Generate transactions
        transactions = [self.generate_transaction() for _ in range(self.num_transactions)]
        
        # Write to CSV (3 columns only)
        with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["date", "merchant", "amount"])
            writer.writeheader()
            writer.writerows(transactions)
        
        print(f"   Generated messy data: {self.output_path}")
        print(f"   Total transactions: {self.num_transactions}")
        print(f"   File size: {self.output_path.stat().st_size:,} bytes")
        
        # Print some statistics
        self._print_statistics(transactions)
        
        return self.output_path
    
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
    generator = MessyDataGenerator()
    csv_path = generator.generate_csv()
    
    print("\n" + "=" * 70)
    print(f"  Success! Generated messy data at:")
    print(f"   {csv_path}")
    print("\n   CSV Format:")
    print("   Columns: date, merchant, amount")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())