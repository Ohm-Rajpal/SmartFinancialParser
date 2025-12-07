"""
Smart Financial Parser CLI

A CLI tool for normalizing messy financial transaction data using AI.

Usage:
    python -m src.cli <num_transactions>    # Generate and process N transactions
    python -m src.cli                       # Interactive: enter number when prompted
"""

import logging
import sys
import time
from typing import Dict

import pandas as pd

import ray

from src.config import Config
from src.pipeline.ray_pipeline import process_financial_data, shutdown_ray
from scripts.generate_messy_data import MessyDataGenerator

# Configure logging for CLI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def display_results(df: pd.DataFrame, metadata: Dict):
    """Display processing results in a formatted table"""
    print("\n" + "=" * 80)
    print("PROCESSING RESULTS")
    print("=" * 80)
    print(f"Total transactions: {metadata.get('total_rows', len(df))}")
    print(f"Processing time: {metadata.get('processing_time_seconds', 0):.2f} seconds")
    print(f"Validation errors: {len(metadata.get('validation_errors', []))}")
    print(f"Unique merchants: {metadata.get('unique_merchants', 0)}")
    print(f"Total amount: ${metadata.get('total_amount', 0):,.2f}")
    
    if metadata.get('top_spending_category'):
        print(f"\nTop Spending Category: {metadata['top_spending_category']}")
        print(f"  Amount: ${metadata.get('top_category_amount', 0):,.2f}")
    
    # Show spending by category
    print("\nSpending by Category:")
    print("-" * 80)
    category_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False) # we want to show the data decreasing order
    
    for category, total in category_totals.items():
        print(f"  {category:30s} $ {total:,.2f}")

    # sample transactions of the top 10
    print("\nSample Transactions (first 10):")
    print("-" * 80)
    print(df.head(10).to_string(index=False))


def generate_and_process(num_transactions: int) -> bool:
    """
    Generate random test data and process it.
    
    Args:
        num_transactions: Number of transactions to generate
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print("Smart Financial Parser")
    print("=" * 80)
    print(f"Generating {num_transactions} random transactions...")
    print("-" * 80)
    
    try:
        # Generate messy data
        generator = MessyDataGenerator(num_transactions=num_transactions)
        csv_path = generator.generate_csv()
        
        print(f"Generated test data: {csv_path}")
        
        # Process the generated data
        print("\nProcessing transactions...")
        df, metadata = process_financial_data(str(csv_path))
        
        # Display results
        display_results(df, metadata)
        
        # Save to clean_transactions.csv, also note that we need to save to S3
        output_path = Config.PROCESSED_DATA_DIR / "clean_transactions.csv"
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n" + "=" * 100)
        print(f"Results saved to: {output_path}")
        print(f"Total rows saved: {len(df)}")
        print("=" * 100 + "\n")
        
        # flush output so frontend does not get interrupted by Ray's logs
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(3.5)
        sys.stdout.flush()
        sys.stderr.flush()
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating/processing data: {e}", exc_info=True)
        print(f"\nError: Failed to generate or process data: {e}")
        return False


# this code is similar to the one inside of merchant_normalizer but is here to speed up the Ray workers
# so they do not have a cold start!
def initialize_ray():
    """Initialize local Ray cluster early for warmup and reuse"""
    if not ray.is_initialized():
        logger.info(f"Initializing local Ray with {Config.RAY_NUM_CPUS} CPUs")
        ray.init(
            num_cpus=Config.RAY_NUM_CPUS, 
            object_store_memory=Config.RAY_OBJECT_STORE_MEMORY, 
            logging_level=logging.INFO
        )
        
        # Warmup Ray workers
        logger.info("Warming up Ray workers...")
        @ray.remote
        def warmup_task():
            return "warmed_up"
        
        # worker creation
        futures = [warmup_task.remote() for _ in range(min(Config.RAY_NUM_CPUS, 10))]
        ray.get(futures)
        logger.info("Ray workers warmed up and ready")
    else:
        logger.info("Ray already initialized, skipping warmup")


def main():
    """Main entry point for CLI"""
    Config.ensure_directories()
    
    # lazy initialization with early warmup of Ray workers
    initialize_ray()
    
    # Check if number of transactions provided as command line argument
    if len(sys.argv) > 1:
        try:
            num_transactions = int(sys.argv[1])
            if num_transactions <= 0:
                print("Error: Number of transactions must be positive")
                shutdown_ray()
                sys.exit(1)
            # Single run mode with command line argument
            try:
                success = generate_and_process(num_transactions)
                shutdown_ray()
                sys.exit(0 if success else 1)
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Shutting down...")
                shutdown_ray()
                sys.exit(1)
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                shutdown_ray()
                sys.exit(1)
        except ValueError:
            print("Error: Number of transactions must be an integer")
            print("Usage: python -m src.cli <num_transactions>")
            shutdown_ray()
            sys.exit(1)
    
    # Interactive loop mode
    print("\n" + "=" * 80)
    print("Smart Financial Parser")
    print("=" * 80)
    print("Interactive Mode")
    print("-" * 80)
    
    while True:
        # flush pending output before showing menu
        sys.stdout.flush()
        sys.stderr.flush()
        
        print("\nOptions:")
        print("  0. Exit")
        print("  1. Generate and process transactions")
        # todo: add more functionality here to include retrieving data from S3

        try:
            choice = input("\nEnter your choice: ").strip()
            
            if choice == "0":
                print("\nExiting. Goodbye!")
                break
            elif choice == "1":
                try:
                    num_str = input("Enter number of transactions to generate: ").strip()
                    num_transactions = int(num_str)
                    
                    if num_transactions <= 0:
                        print("Error: Number of transactions must be positive")
                        continue
                    
                    success = generate_and_process(num_transactions)
                    if not success:
                        print("Error: Processing failed. Please try again.")
                    
                except ValueError:
                    print("Error: Invalid number. Please enter a positive integer.")
                except KeyboardInterrupt:
                    print("\n\nInterrupted by user. Returning to menu...")
                    continue
            else:
                print("Error: Invalid choice. Please enter 1 or 0.")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            print(f"Error: {e}. Please try again.")
    
    # Cleanup on exit
    shutdown_ray()

if __name__ == '__main__':
    main()
