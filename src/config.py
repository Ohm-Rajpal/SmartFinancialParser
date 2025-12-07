"""
Central configuration for Financial Parser CLI.

This module contains all application settings, paths, and constants.
All other modules import configuration from here to maintain consistency.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration and constants"""
    
    # ==========================================
    # Project Paths
    # ==========================================
    PROJECT_ROOT = Path(__file__).parent.parent
    SRC_DIR = PROJECT_ROOT / "src"
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
    SCRIPTS_DIR = PROJECT_ROOT / "scripts"
    DOCS_DIR = PROJECT_ROOT / "docs"
    TESTS_DIR = PROJECT_ROOT / "tests"
    
    # ==========================================
    # AWS Configuration
    # ==========================================
    AWS_REGION = os.getenv("AWS_REGION", "us-west-1")
    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "financial-parser-ohm-2025")
    
    # S3 path for storing analysis
    S3_ANALYSES_PREFIX = "analyses"  # s3://bucket/analyses/
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Flag to indicate if using local secrets (for logging warnings)
    USING_LOCAL_SECRETS = bool(OPENAI_API_KEY or GEMINI_API_KEY)
    
    # ==========================================
    # Ray Configuration
    # ==========================================
    RAY_NUM_CPUS = int(os.getenv("RAY_NUM_CPUS"))
    RAY_OBJECT_STORE_MEMORY = int(os.getenv("RAY_OBJECT_STORE_MEMORY"))
    RAY_ENABLE_LOGGING = os.getenv("RAY_ENABLE_LOGGING", "true").lower() == "true"
    
    # ==========================================
    # Application Settings
    # ==========================================
    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")          # DEBUG, INFO, WARNING, ERROR
    MAX_RETRIES = int(os.getenv("MAX_RETRIES"))         # API call retries
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT")) # Timeout
    
    # ==========================================
    # LLM Configuration
    # ==========================================
    GPT_MODEL = os.getenv("GPT_MODEL")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL",)
    
    # LLM parameters
    TEMPERATURE = float(os.getenv("TEMPERATURE"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
    
    # Rate limiting (requests per minute)
    GPT_RATE_LIMIT = int(os.getenv("GPT_RATE_LIMIT"))
    GEMINI_RATE_LIMIT = int(os.getenv("GEMINI_RATE_LIMIT"))
    
    # ==========================================
    # Merchant Categories
    # ==========================================
    # Standardized categories for merchant classification
    MERCHANT_CATEGORIES: List[str] = [
        "Food & Dining",
        "Transportation",
        "Shopping",
        "Entertainment",
        "Bills & Utilities",
        "Healthcare",
        "Travel",
        "Personal Care",
        "Education",
        "Other"
    ]
    
    # fallback category classification if LLM fails
    CATEGORY_KEYWORDS = {
        "Food & Dining": [
            "restaurant", "cafe", "coffee", "starbucks", "mcdonalds",
            "burger", "pizza", "food", "dining", "grocery", "market"
        ],
        "Transportation": [
            "uber", "lyft", "taxi", "gas", "fuel", "parking", "metro",
            "transit", "bus", "train", "airline", "flight"
        ],
        "Shopping": [
            "amazon", "walmart", "target", "store", "shop", "retail",
            "clothing", "fashion", "electronics"
        ],
        "Entertainment": [
            "movie", "cinema", "theater", "netflix", "spotify", "game",
            "concert", "event", "ticket"
        ],
        "Bills & Utilities": [
            "electric", "water", "gas", "internet", "phone", "utility",
            "bill", "insurance", "rent", "mortgage"
        ],
        "Healthcare": [
            "pharmacy", "doctor", "hospital", "clinic", "medical",
            "health", "cvs", "walgreens", "dental"
        ],
        "Travel": [
            "hotel", "airbnb", "booking", "vacation", "resort",
            "expedia", "travel"
        ],
        "Personal Care": [
            "salon", "spa", "gym", "fitness", "haircut", "beauty"
        ],
        "Education": [
            "school", "university", "college", "course", "tuition",
            "book", "education"
        ]
    }
    
    # ==========================================
    # Database Configuration
    # ==========================================
    # to display data 
    DB_PATH = PROJECT_ROOT / "financial_parser.db"
    
    # SQLite table schemas
    ANALYSES_TABLE_SCHEMA = """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            filename TEXT NOT NULL,
            s3_path TEXT,
            total_transactions INTEGER,
            total_amount REAL,
            top_category TEXT,
            top_category_amount REAL,
            gpt_model TEXT,
            gemini_model TEXT,
            processing_time_seconds REAL,
            accuracy_score REAL
        )
    """
    
    MERCHANTS_CACHE_TABLE_SCHEMA = """
        CREATE TABLE IF NOT EXISTS merchant_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_name TEXT UNIQUE NOT NULL,
            normalized_name TEXT NOT NULL,
            category TEXT NOT NULL,
            source_model TEXT,
            confidence REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    
    # ==========================================
    # Security Settings
    # ==========================================
    # Encryption settings
    ENCRYPTION_ALGORITHM = "AES256"
    
    # Dangerous CSV injection prefixes
    CSV_INJECTION_PREFIXES = ["=", "+", "-", "@", "\t", "\r"]
    
    # Prompt injection patterns to detect
    PROMPT_INJECTION_PATTERNS = [
        r"ignore previous instructions",
        r"ignore all previous",
        r"disregard",
        r"system:",
        r"</system>",
        r"admin",
        r"root",
        r"<script",
        r"javascript:",
        r"exec\s*\(",
        r"eval\s*\("
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.",
        r"/etc/",
        r"/root/",
        r"~/"
    ]
    
    # ==========================================
    # Data Validation Rules
    # ==========================================
    # Date validation
    MIN_TRANSACTION_YEAR = 2000  # Reject dates before this
    MAX_TRANSACTION_YEAR = 2025  # Reject dates after this
    
    # Amount validation
    MIN_TRANSACTION_AMOUNT = -100000.0  # Large refunds allowed
    MAX_TRANSACTION_AMOUNT = 100000.0   # Prevent unrealistic transactions
    
    # Merchant validation
    MAX_MERCHANT_NAME_LENGTH = 200
    MIN_MERCHANT_NAME_LENGTH = 1
    
    # ==========================================
    # Test Data Settings
    # ==========================================
    # For generate_messy_data.py script
    NUM_TEST_TRANSACTIONS = 500  # Number of rows to generate
    
    # Date formats to mix in messy data
    DATE_FORMATS = [
        "%Y-%m-%d",           # 2023-01-01
        "%m/%d/%Y",           # 01/01/2023
        "%d/%m/%Y",           # 01/01/2023
        "%B %d, %Y",          # January 01, 2023
        "%b %d %Y",           # Jan 01 2023
        "%d %b %Y",           # 01 Jan 2023
        "%Y/%m/%d",           # 2023/01/01
        "%m-%d-%Y",           # 01-01-2023
        "%d.%m.%Y",           # 01.01.2023
        "%Y%m%d"              # 20230101
    ]
    
    # Currency symbols to mix in amounts
    CURRENCY_SYMBOLS = ["$", "USD", "€", "EUR", "£", "GBP"]
    
    # ==========================================
    # Logging Configuration
    # ==========================================
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Log file paths
    LOGS_DIR = PROJECT_ROOT / "logs"
    MAIN_LOG_FILE = LOGS_DIR / "financial_parser.log"
    ERROR_LOG_FILE = LOGS_DIR / "errors.log"
    MODEL_COMPARISON_LOG = LOGS_DIR / "model_comparison.log"
    
    # ==========================================
    # Performance Monitoring
    # ==========================================
    ENABLE_PERFORMANCE_LOGGING = True
    PERFORMANCE_LOG_FILE = LOGS_DIR / "performance.log"
    
    # ==========================================
    # Output Formatting
    # ==========================================
    # CLI output settings
    CLI_MAX_WIDTH = 120
    CLI_SHOW_PROGRESS_BAR = True
    
    # JSON output settings
    JSON_INDENT = 2
    JSON_ENSURE_ASCII = False
    
    # ==========================================
    # Class Methods
    # ==========================================
    @classmethod
    def ensure_directories(cls) -> None:
        """
        Create all necessary directories if they don't exist.
        
        This should be called at application startup to ensure
        the file system is properly initialized.
        """
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.GROUND_TRUTH_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_environment(cls) -> bool:
        """
        Validate that all required environment variables are set.
        
        Returns:
            bool: True if environment is valid, False otherwise
        """
        required_vars = [
            ("AWS_REGION", cls.AWS_REGION),
            ("AWS_S3_BUCKET", cls.AWS_S3_BUCKET)
        ]
        
        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            print(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False
        
        return True
    
    @classmethod
    def get_s3_path(cls, filename: str) -> str:
        """
        Generate S3 path for a file.
        
        Returns:
            str: Full S3 path (e.g., "s3://bucket/analyses/file.json")
        """
        return f"s3://{cls.AWS_S3_BUCKET}/{cls.S3_ANALYSES_PREFIX}/{filename}"
    
    @classmethod
    def is_development_mode(cls) -> bool:
        """
        Check if running in development mode (using local secrets).
        
        Returns:
            bool: True if in development mode
        """
        return cls.USING_LOCAL_SECRETS
    
    @classmethod
    def print_config_summary(cls) -> None:
        """Print configuration summary for debugging"""
        print("=" * 60)
        print("Financial Parser - Configuration Summary")
        print("=" * 60)
        print(f"AWS Region:           {cls.AWS_REGION}")
        print(f"S3 Bucket:            {cls.AWS_S3_BUCKET}")
        print(f"GPT Model:            {cls.GPT_MODEL}")
        print(f"Gemini Model:         {cls.GEMINI_MODEL}")
        print(f"Batch Size:           {cls.BATCH_SIZE}")
        print(f"Ray CPUs:             {cls.RAY_NUM_CPUS}")
        print(f"Development Mode:     {cls.is_development_mode()}")
        print(f"Database:             {cls.DB_PATH}")
        print(f"Categories:           {len(cls.MERCHANT_CATEGORIES)}")
        print("=" * 60)


# Initialize directories on import
Config.ensure_directories()
