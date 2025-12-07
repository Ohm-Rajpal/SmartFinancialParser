"""
S3 handler for storing and retrieving financial analysis results.

Provides functionality to:
- Upload processed results to S3
- List available analyses
- Download and view past analyses
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from src.config import Config

logger = logging.getLogger(__name__)


class S3Handler:
    """Handles S3 operations for financial analysis storage"""
    
    def __init__(self):
        """Initialize S3 client"""
        try:
            self.s3_client = boto3.client('s3', region_name=Config.AWS_REGION)
            self.bucket_name = Config.AWS_S3_BUCKET
            self.prefix = Config.S3_ANALYSES_PREFIX
            logger.info(f"S3 handler initialized for bucket: {self.bucket_name}")
        except NoCredentialsError:
            logger.warning("AWS credentials not found. S3 operations will fail.")
            self.s3_client = None
    
    def upload_analysis(self, df, metadata: Dict, filename: Optional[str] = None) -> Optional[str]:
        """
        Upload processed analysis to S3.
        
        Args:
            df: Processed DataFrame
            metadata: Processing metadata
            filename: Optional custom filename (default: timestamp-based)
            
        Returns:
            S3 key if successful, None otherwise
        """
        if not self.s3_client:
            logger.error("S3 client not initialized. Cannot upload.")
            return None
        
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"analysis_{timestamp}.json"
            
            s3_key = f"{self.prefix}/{filename}"
            
            # Prepare data for JSON serialization
            analysis_data = {
                "metadata": metadata,
                "transactions": df.to_dict('records'),
                "summary": {
                    "total_transactions": len(df),
                    "total_amount": float(df['amount'].sum()),
                    "top_category": metadata.get('top_spending_category', 'Unknown'),
                    "top_category_amount": float(metadata.get('top_category_amount', 0)),
                    "unique_merchants": metadata.get('unique_merchants', 0),
                    "processing_time": metadata.get('processing_time_seconds', 0)
                }
            }
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json.dumps(analysis_data, indent=2, default=str),
                ContentType='application/json'
            )
            
            logger.info(f"Analysis uploaded to s3://{self.bucket_name}/{s3_key}")
            return s3_key
            
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading to S3: {e}")
            return None
    
    def list_analyses(self) -> List[Dict]:
        """
        List all available analyses in S3.
        
        Returns:
            List of analysis metadata dictionaries with keys:
            - key: S3 key
            - filename: Filename
            - last_modified: Last modified timestamp
            - size: File size in bytes
        """
        if not self.s3_client:
            logger.error("S3 client not initialized. Cannot list analyses.")
            return []
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{self.prefix}/"
            )
            
            if 'Contents' not in response:
                return []
            
            analyses = []
            for obj in response['Contents']:
                key = obj['Key']
                filename = Path(key).name
                
                analyses.append({
                    'key': key,
                    'filename': filename,
                    'last_modified': obj['LastModified'],
                    'size': obj['Size']
                })
            
            # Sort by last modified (newest first)
            analyses.sort(key=lambda x: x['last_modified'], reverse=True)
            
            return analyses
            
        except ClientError as e:
            logger.error(f"Failed to list S3 objects: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing S3 objects: {e}")
            return []
    
    def download_analysis(self, s3_key: str) -> Optional[Dict]:
        """
        Download and parse an analysis from S3.
        
        Args:
            s3_key: S3 key of the analysis file
            
        Returns:
            Parsed analysis data dictionary, or None if failed
        """
        if not self.s3_client:
            logger.error("S3 client not initialized. Cannot download.")
            return None
        
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            content = response['Body'].read().decode('utf-8')
            analysis_data = json.loads(content)
            
            logger.info(f"Downloaded analysis from s3://{self.bucket_name}/{s3_key}")
            return analysis_data
            
        except ClientError as e:
            logger.error(f"Failed to download from S3: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from S3: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading from S3: {e}")
            return None
    
    def delete_analysis(self, s3_key: str) -> bool:
        """
        Delete an analysis from S3.
        
        Args:
            s3_key: S3 key of the analysis file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.s3_client:
            logger.error("S3 client not initialized. Cannot delete.")
            return False
        
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            logger.info(f"Deleted analysis from s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete from S3: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting from S3: {e}")
            return False

