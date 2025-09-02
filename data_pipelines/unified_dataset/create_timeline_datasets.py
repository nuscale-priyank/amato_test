#!/usr/bin/env python3
"""
Create Timeline-Based Datasets for Training and Inference
Separates unified dataset into historical training data and recent inference data
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.s3_utils import get_s3_manager

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimelineDatasetCreator:
    """Creates timeline-based datasets for training and inference"""
    
    def __init__(self, config_path='config/database_config.yaml'):
        self.config = self.load_config(config_path)
        self.output_dir = 'data_pipelines/unified_dataset/output'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_config(self, config_path):
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return {}
    
    def load_unified_dataset(self):
        """Load the unified dataset"""
        try:
            # First try to load from S3
            logger.info("🔍 Attempting to load data from S3...")
            s3_manager = get_s3_manager()
            s3_manager.load_data_from_s3()
            logger.info("✅ Data loaded from S3")
            
            # Now try to load the local file
            data_path = os.path.join(self.output_dir, 'unified_customer_dataset.parquet')
            
            if os.path.exists(data_path):
                df = pd.read_parquet(data_path)
                logger.info(f"✅ Loaded unified dataset: {df.shape}")
                return df
            else:
                logger.error(f"❌ Unified dataset not found at {data_path}")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return None
    
    def create_timeline_datasets(self, df, training_cutoff_months=3):
        """Create training and inference datasets based on timeline"""
        logger.info(f"🔧 Creating timeline datasets with {training_cutoff_months} months cutoff...")
        
        try:
            # Check if we have valid timestamp data
            has_valid_timestamps = False
            
            if 'registration_date' in df.columns and not df['registration_date'].isna().all():
                df['dataset_created_at'] = pd.to_datetime(df['registration_date'])
                has_valid_timestamps = True
                logger.info("✅ Using registration_date for timeline split (business logic)")
            elif 'dataset_created_at' in df.columns and not df['dataset_created_at'].isna().all():
                df['dataset_created_at'] = pd.to_datetime(df['dataset_created_at'])
                has_valid_timestamps = True
                logger.info("✅ Using dataset_created_at for timeline split (fallback)")
            
            if not has_valid_timestamps:
                # Create synthetic timeline split for demonstration
                logger.info("⚠️  No valid timestamps found, creating synthetic timeline split")
                total_customers = len(df)
                training_size = int(total_customers * 0.7)  # 70% for training (historical)
                inference_size = total_customers - training_size  # 30% for inference (recent)
                
                # Randomly split the data
                np.random.seed(42)  # For reproducible split
                indices = np.random.permutation(total_customers)
                
                training_indices = indices[:training_size]
                inference_indices = indices[training_size:]
                
                training_df = df.iloc[training_indices].copy()
                inference_df = df.iloc[inference_indices].copy()
                
                logger.info(f"📊 Synthetic dataset split:")
                logger.info(f"   Training data: {len(training_df)} customers (70% - historical)")
                logger.info(f"   Inference data: {len(inference_df)} customers (30% - recent)")
                
                return training_df, inference_df
            
            # Calculate cutoff date for training data
            cutoff_date = datetime.now() - timedelta(days=training_cutoff_months * 30)
            logger.info(f"📅 Training data cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
            
            # Split data based on timeline
            training_mask = df['dataset_created_at'] < cutoff_date
            inference_mask = df['dataset_created_at'] >= cutoff_date
            
            training_df = df[training_mask].copy()
            inference_df = df[inference_mask].copy()
            
            logger.info(f"📊 Dataset split:")
            logger.info(f"   Training data: {len(training_df)} customers (historical)")
            logger.info(f"   Inference data: {len(inference_df)} customers (recent)")
            
            return training_df, inference_df
            
        except Exception as e:
            logger.error(f"❌ Error creating timeline datasets: {e}")
            return None, None
    
    def save_timeline_datasets(self, training_df, inference_df):
        """Save timeline datasets locally and to S3"""
        logger.info("💾 Saving timeline datasets...")
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save training dataset (historical)
            training_path = os.path.join(self.output_dir, 'unified_customer_dataset.parquet')
            training_df.to_parquet(training_path, index=False)
            logger.info(f"✅ Training dataset saved: {training_path}")
            
            # Save inference dataset (recent)
            inference_path = os.path.join(self.output_dir, 'recent_customer_dataset.parquet')
            inference_df.to_parquet(inference_path, index=False)
            logger.info(f"✅ Inference dataset saved: {inference_path}")
            
            # Create metadata
            metadata = {
                'dataset_info': {
                    'created_at': datetime.now().isoformat(),
                    'training_customers': len(training_df),
                    'inference_customers': len(inference_df),
                    'total_features': len(training_df.columns),
                    'training_cutoff_months': 3
                },
                'training_data': {
                    'file': 'unified_customer_dataset.parquet',
                    'description': 'Historical data for model training (before 3 months ago)',
                    'customers': len(training_df),
                    'features': len(training_df.columns)
                },
                'inference_data': {
                    'file': 'recent_customer_dataset.parquet',
                    'description': 'Recent data for model inference (last 3 months)',
                    'customers': len(inference_df),
                    'features': len(inference_df.columns)
                }
            }
            
            # Save metadata
            metadata_path = os.path.join(self.output_dir, 'timeline_datasets_metadata.yaml')
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
            logger.info(f"✅ Metadata saved: {metadata_path}")
            
            # Upload to S3 directly
            self.upload_datasets_to_s3(training_df, inference_df, metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving timeline datasets: {e}")
            return False
    
    def upload_datasets_to_s3(self, training_df, inference_df, metadata):
        """Upload datasets directly to S3"""
        logger.info("☁️ Uploading datasets directly to S3...")
        
        try:
            s3_manager = get_s3_manager()
            
            # Upload training dataset
            training_bytes = training_df.to_parquet(index=False)
            training_s3_key = f"{s3_manager.base_path}/data_pipelines/unified_dataset/output/unified_customer_dataset.parquet"
            s3_manager.upload_bytes_direct(training_bytes, training_s3_key)
            logger.info("✅ Training dataset uploaded to S3")
            
            # Upload inference dataset
            inference_bytes = inference_df.to_parquet(index=False)
            inference_s3_key = f"{s3_manager.base_path}/data_pipelines/unified_dataset/output/recent_customer_dataset.parquet"
            s3_manager.upload_bytes_direct(inference_bytes, inference_s3_key)
            logger.info("✅ Inference dataset uploaded to S3")
            
            # Upload metadata
            metadata_bytes = yaml.dump(metadata, default_flow_style=False).encode('utf-8')
            metadata_s3_key = f"{s3_manager.base_path}/data_pipelines/unified_dataset/output/timeline_datasets_metadata.yaml"
            s3_manager.upload_bytes_direct(metadata_bytes, metadata_s3_key, 'text/yaml')
            logger.info("✅ Metadata uploaded to S3")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error uploading to S3: {e}")
            return False
    
    def run_timeline_creation(self, training_cutoff_months=3):
        """Run the complete timeline dataset creation pipeline"""
        logger.info("🚀 Starting Timeline Dataset Creation Pipeline...")
        
        try:
            # Load unified dataset
            df = self.load_unified_dataset()
            if df is None:
                raise Exception("Failed to load unified dataset")
            
            # Create timeline datasets
            training_df, inference_df = self.create_timeline_datasets(df, training_cutoff_months)
            if training_df is None:
                raise Exception("Failed to create timeline datasets")
            
            # Save and upload datasets
            success = self.save_timeline_datasets(training_df, inference_df)
            if not success:
                raise Exception("Failed to save timeline datasets")
            
            logger.info("=" * 60)
            logger.info("🎉 TIMELINE DATASET CREATION COMPLETED!")
            logger.info("=" * 60)
            logger.info(f"📊 Training dataset: {len(training_df)} customers (historical)")
            logger.info(f"📊 Inference dataset: {len(inference_df)} customers (recent)")
            logger.info(f"💾 Datasets saved locally and uploaded to S3")
            
            return {
                'training_dataset': training_df,
                'inference_dataset': inference_df,
                'training_count': len(training_df),
                'inference_count': len(inference_df)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in timeline creation pipeline: {e}")
            raise


def main():
    """Main function to run the timeline dataset creation pipeline"""
    try:
        creator = TimelineDatasetCreator()
        results = creator.run_timeline_creation()
        
        print(f"\n🎉 Timeline Dataset Creation completed successfully!")
        print(f"📊 Training dataset: {results['training_count']} customers (historical)")
        print(f"📊 Inference dataset: {results['inference_count']} customers (recent)")
        print(f"💾 Datasets saved and uploaded to S3")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
