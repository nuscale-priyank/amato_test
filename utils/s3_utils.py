#!/usr/bin/env python3
"""
S3 Utilities for AMATO Project
Handles uploading, downloading, and syncing files with S3 storage
"""

import os
import yaml
import boto3
import logging
import joblib
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class S3Manager:
    """Manages S3 operations for the AMATO project"""
    
    def __init__(self, config_path: str = "config/s3_config.yaml"):
        """Initialize S3 manager with configuration"""
        # Auto-detect config path based on execution context
        if config_path == "config/s3_config.yaml":
            # Try to find the config file relative to current working directory
            current_dir = Path.cwd()
            if (current_dir / "config" / "s3_config.yaml").exists():
                config_path = str(current_dir / "config" / "s3_config.yaml")
            elif (current_dir.parent / "config" / "s3_config.yaml").exists():
                config_path = str(current_dir.parent / "config" / "s3_config.yaml")
            elif (current_dir.parent.parent / "config" / "s3_config.yaml").exists():
                config_path = str(current_dir.parent.parent / "config" / "s3_config.yaml")
            else:
                # Fallback to default
                config_path = "config/s3_config.yaml"
        
        self.config = self._load_config(config_path)
        self.s3_client = boto3.client('s3')
        self.bucket = self.config['s3']['bucket']
        self.base_path = self.config['s3']['base_path']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load S3 configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"S3 config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing S3 config: {e}")
            raise
    
    def _get_s3_path(self, local_path: str, s3_subpath: str) -> str:
        """Generate S3 path from local path and subpath"""
        return f"{self.base_path}/{s3_subpath}"
    
    def upload_file(self, local_path: str, s3_subpath: str, overwrite: bool = True) -> bool:
        """Upload a single file to S3"""
        try:
            s3_path = self._get_s3_path(local_path, s3_subpath)
            s3_key = f"{s3_path}/{os.path.basename(local_path)}"
            
            if not overwrite and self._file_exists(s3_key):
                logger.info(f"File already exists in S3: {s3_key}")
                return True
            
            logger.info(f"Uploading {local_path} to s3://{self.bucket}/{s3_key}")
            self.s3_client.upload_file(local_path, self.bucket, s3_key)
            logger.info(f"Successfully uploaded {local_path}")
            return True
            
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Error uploading {local_path}: {e}")
            return False
    
    def download_file(self, s3_key: str, local_path: str, overwrite: bool = True) -> bool:
        """Download a single file from S3"""
        try:
            if not overwrite and os.path.exists(local_path):
                logger.info(f"File already exists locally: {local_path}")
                return True
            
            logger.info(f"Downloading s3://{self.bucket}/{s3_key} to {local_path}")
            self.s3_client.download_file(self.bucket, s3_key, local_path)
            logger.info(f"Successfully downloaded {s3_key}")
            return True
            
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Error downloading {s3_key}: {e}")
            return False
    
    def upload_directory(self, local_dir: str, s3_subpath: str, 
                        file_types: Optional[List[str]] = None) -> Dict[str, bool]:
        """Upload all files in a directory to S3"""
        results = {}
        local_path = Path(local_dir)
        
        if not local_path.exists():
            logger.error(f"Local directory does not exist: {local_dir}")
            return results
        
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                # Check file type filter
                if file_types and not any(file_path.suffix.lower() in ft for ft in file_types):
                    continue
                
                # Calculate relative path for S3
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{self.base_path}/{s3_subpath}/{relative_path}"
                
                try:
                    logger.info(f"Uploading {file_path} to s3://{self.bucket}/{s3_key}")
                    self.s3_client.upload_file(str(file_path), self.bucket, str(s3_key))
                    results[str(file_path)] = True
                except (ClientError, NoCredentialsError) as e:
                    logger.error(f"Error uploading {file_path}: {e}")
                    results[str(file_path)] = False
        
        return results
    
    def download_directory(self, s3_prefix: str, local_dir: str, 
                          file_types: Optional[List[str]] = None) -> Dict[str, bool]:
        """Download all files from S3 prefix to local directory"""
        results = {}
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # List objects in S3 prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    
                    # Check file type filter
                    if file_types and not any(s3_key.lower().endswith(ft) for ft in file_types):
                        continue
                    
                    # Calculate local path - flatten to avoid nested directories
                    filename = os.path.basename(s3_key)
                    local_file_path = local_path / filename
                    
                    try:
                        logger.info(f"Downloading s3://{self.bucket}/{s3_key} to {local_file_path}")
                        self.s3_client.download_file(self.bucket, s3_key, str(local_file_path))
                        results[s3_key] = True
                    except (ClientError, NoCredentialsError) as e:
                        logger.error(f"Error downloading {s3_key}: {e}")
                        results[s3_key] = False
                        
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Error listing S3 objects: {e}")
        
        return results
    
    def _file_exists(self, s3_key: str) -> bool:
        """Check if a file exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False
    
    def list_files(self, s3_prefix: str) -> List[str]:
        """List all files in S3 prefix"""
        files = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix)
            
            for page in pages:
                if 'Contents' in page:
                    files.extend([obj['Key'] for obj in page['Contents']])
                    
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Error listing S3 files: {e}")
        
        return files

    def list_files_with_meta(self, s3_prefix: str) -> List[Dict[str, str]]:
        """List files with metadata (Key, LastModified, Size) in S3 prefix"""
        objects: List[Dict[str, str]] = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix)
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append({
                            'Key': obj['Key'],
                            'LastModified': obj.get('LastModified'),
                            'Size': obj.get('Size')
                        })
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Error listing S3 objects: {e}")
        return objects

    def download_latest_by_suffix(self, s3_subdir: str, local_dir: str, suffixes: List[str]) -> Dict[str, Optional[str]]:
        """Download the latest file for each suffix under a subdir. Returns mapping suffix->local_path."""
        results: Dict[str, Optional[str]] = {suffix: None for suffix in suffixes}
        prefix = f"{self.base_path}/{s3_subdir}".rstrip('/') + '/'
        objects = self.list_files_with_meta(prefix)
        if not objects:
            logger.warning(f"No objects found under {prefix}")
            return results
        local_path_obj = Path(local_dir)
        local_path_obj.mkdir(parents=True, exist_ok=True)

        for suffix in suffixes:
            # Filter by suffix and choose latest by LastModified
            candidates = [o for o in objects if o['Key'].lower().endswith(suffix.lower())]
            if not candidates:
                continue
            latest = max(candidates, key=lambda o: o.get('LastModified'))
            key = latest['Key']
            relative = key.replace(f"{self.base_path}/", "")
            local_file = local_path_obj / relative
            local_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                logger.info(f"Downloading latest {suffix} from {key} -> {local_file}")
                self.s3_client.download_file(self.bucket, key, str(local_file))
                results[suffix] = str(local_file)
            except (ClientError, NoCredentialsError) as e:
                logger.error(f"Error downloading latest {suffix} from {key}: {e}")
        return results
    
    def delete_file(self, s3_key: str) -> bool:
        """Delete a file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
            logger.info(f"Deleted s3://{self.bucket}/{s3_key}")
            return True
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Error deleting {s3_key}: {e}")
            return False
    
    def sync_models_to_s3(self, local_models_dir: str = "models") -> Dict[str, bool]:
        """Sync all model files to S3"""
        logger.info("Syncing models to S3...")
        return self.upload_directory(
            local_models_dir, 
            "models/", 
            file_types=self.config['file_types']['models']
        )
    
    def sync_data_to_s3(self, local_data_dir: str = "data_pipelines/unified_dataset/output") -> Dict[str, bool]:
        """Sync data pipeline outputs to S3"""
        logger.info("Syncing data pipeline outputs to S3...")
        return self.upload_directory(
            local_data_dir, 
            "data_pipelines/unified_dataset/output/", 
            file_types=self.config['file_types']['data']
        )
    
    def sync_inference_results_to_s3(self, local_results_dir: str = "models") -> Dict[str, bool]:
        """Sync inference results to S3"""
        logger.info("Syncing inference results to S3...")
        results = {}
        
        # Sync each model type's inference results
        for model_type in ['campaign_optimization', 'customer_segmentation', 'forecasting', 'journey_simulation']:
            local_path = f"{local_results_dir}/{model_type}/inference_results"
            if os.path.exists(local_path):
                s3_subpath = f"models/{model_type}/inference_results/"
                model_results = self.upload_directory(
                    local_path, 
                    s3_subpath, 
                    file_types=self.config['file_types']['reports'] + self.config['file_types']['visualizations']
                )
                results.update(model_results)
        
        return results
    
    def load_models_from_s3(self, local_models_dir: str = "models") -> Dict[str, bool]:
        """Load all model files from S3"""
        logger.info("Loading models from S3...")
        return self.download_directory(
            f"{self.base_path}/models/", 
            local_models_dir, 
            file_types=self.config['file_types']['models']
        )
    
    def load_data_from_s3(self, local_data_dir: str = "data_pipelines/unified_dataset/output") -> Dict[str, bool]:
        """Load data pipeline outputs from S3"""
        logger.info("Loading data pipeline outputs from S3...")
        return self.download_directory(
            f"{self.base_path}/data_pipelines/unified_dataset/output/", 
            local_data_dir, 
            file_types=self.config['file_types']['data']
        )
    
    def upload_bytes_direct(self, data: bytes, s3_key: str, content_type: str = None) -> bool:
        """Upload bytes directly to S3 without local file creation"""
        try:
            if content_type is None:
                # Auto-detect content type from file extension
                if s3_key.endswith('.pkl'):
                    content_type = 'application/octet-stream'
                elif s3_key.endswith('.yaml') or s3_key.endswith('.yml'):
                    content_type = 'text/yaml'
                elif s3_key.endswith('.json'):
                    content_type = 'application/json'
                elif s3_key.endswith('.parquet'):
                    content_type = 'application/octet-stream'
                else:
                    content_type = 'application/octet-stream'
            
            logger.info(f"Uploading bytes directly to s3://{self.bucket}/{s3_key}")
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=data,
                ContentType=content_type
            )
            logger.info(f"Successfully uploaded bytes to s3://{self.bucket}/{s3_key}")
            return True
            
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Error uploading bytes to {s3_key}: {e}")
            return False
    
    def upload_model_direct(self, model, model_name: str, model_type: str, metadata: Dict = None) -> bool:
        """Upload model directly to S3 without local storage"""
        try:
            # Serialize model to bytes using BytesIO
            import io
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            model_bytes = buffer.getvalue()
            
            # Create S3 key
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"{self.base_path}/models/{model_type}/{model_name}_{timestamp}.pkl"
            
            # Upload model bytes
            success = self.upload_bytes_direct(model_bytes, s3_key)
            
            if success and metadata:
                # Upload metadata
                metadata_bytes = yaml.dump(metadata, default_flow_style=False).encode('utf-8')
                metadata_key = f"{self.base_path}/models/{model_type}/{model_name}_metadata_{timestamp}.yaml"
                self.upload_bytes_direct(metadata_bytes, metadata_key, 'text/yaml')
            
            return success
            
        except Exception as e:
            logger.error(f"Error uploading model {model_name}: {e}")
            return False
    
    def load_training_data_from_s3(self, local_data_dir: str = "data_pipelines/unified_dataset/output") -> Dict[str, bool]:
        """Load historical training data from S3 (anything before 3 months ago)"""
        logger.info("Loading historical training data from S3...")
        return self.download_directory(
            f"{self.base_path}/data_pipelines/unified_dataset/output/", 
            local_data_dir, 
            file_types=self.config['file_types']['data']
        )
    
    def load_inference_data_from_s3(self, months_recent: int = 3, local_data_dir: str = "data_pipelines/unified_dataset/output") -> Dict[str, bool]:
        """Load recent inference data from S3 (last 1, 2, or 3 months)"""
        logger.info(f"Loading recent inference data from S3 (last {months_recent} months)...")
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=months_recent * 30)
        cutoff_timestamp = cutoff_date.strftime('%Y-%m-%d')
        
        logger.info(f"Loading data newer than {cutoff_timestamp}")
        
        # For now, download all data and filter locally
        # In production, this could be optimized with S3 Select or Athena queries
        return self.download_directory(
            f"{self.base_path}/data_pipelines/unified_dataset/output/", 
            local_data_dir, 
            file_types=self.config['file_types']['data']
        )
    
    def get_data_timeline_info(self) -> Dict[str, str]:
        """Get information about available data timelines"""
        try:
            # List available datasets and their timestamps
            datasets = self.list_files_with_meta(f"{self.base_path}/data_pipelines/unified_dataset/output/")
            
            timeline_info = {
                'training_data': 'unified_customer_dataset.parquet',  # Historical data
                'inference_data': 'recent_customer_dataset.parquet',  # Recent data (configurable)
                'available_timestamps': []
            }
            
            for dataset in datasets:
                if dataset['Key'].endswith('.parquet'):
                    timeline_info['available_timestamps'].append({
                        'file': dataset['Key'],
                        'last_modified': dataset['LastModified'],
                        'size': dataset['Size']
                    })
            
            return timeline_info
            
        except Exception as e:
            logger.error(f"Error getting data timeline info: {e}")
            return {}


def get_s3_manager(config_path: str = "config/s3_config.yaml") -> S3Manager:
    """Factory function to get S3 manager instance"""
    # Auto-detect config path if not explicitly provided
    if config_path == "config/s3_config.yaml":
        # Try to find the config file relative to current working directory
        current_dir = Path.cwd()
        if (current_dir / "config" / "s3_config.yaml").exists():
            config_path = str(current_dir / "config" / "s3_config.yaml")
        elif (current_dir.parent / "config" / "s3_config.yaml").exists():
            config_path = str(current_dir.parent / "config" / "s3_config.yaml")
        elif (current_dir.parent.parent / "config" / "s3_config.yaml").exists():
            config_path = str(current_dir.parent.parent / "config" / "s3_config.yaml")
        else:
            # Fallback to default
            config_path = "config/s3_config.yaml"
    
    return S3Manager(config_path)


# Convenience functions for common operations
def upload_model_file(local_path: str, model_type: str) -> bool:
    """Upload a single model file to S3"""
    s3_manager = get_s3_manager()
    s3_subpath = f"models/{model_type}/"
    return s3_manager.upload_file(local_path, s3_subpath)


def download_model_file(s3_key: str, local_path: str) -> bool:
    """Download a single model file from S3"""
    s3_manager = get_s3_manager()
    return s3_manager.download_file(s3_key, local_path)


def sync_all_to_s3() -> Dict[str, Dict[str, bool]]:
    """Sync all AMATO files to S3"""
    s3_manager = get_s3_manager()
    results = {
        'models': s3_manager.sync_models_to_s3(),
        'data': s3_manager.sync_data_to_s3(),
        'inference_results': s3_manager.sync_inference_results_to_s3()
    }
    return results


def load_all_from_s3() -> Dict[str, Dict[str, bool]]:
    """Load all AMATO files from S3"""
    s3_manager = get_s3_manager()
    results = {
        'models': s3_manager.load_models_from_s3(),
        'data': s3_manager.load_data_from_s3()
    }
    return results


def upload_model_direct(model, model_name: str, model_type: str, metadata: Dict = None) -> bool:
    """Upload model directly to S3 without local storage"""
    s3_manager = get_s3_manager()
    return s3_manager.upload_model_direct(model, model_name, model_type, metadata)


def load_training_data_from_s3(local_data_dir: str = "data_pipelines/unified_dataset/output") -> Dict[str, bool]:
    """Load historical training data from S3 (anything before 3 months ago)"""
    s3_manager = get_s3_manager()
    return s3_manager.load_training_data_from_s3(local_data_dir)


def load_inference_data_from_s3(months_recent: int = 3, local_data_dir: str = "data_pipelines/unified_dataset/output") -> Dict[str, bool]:
    """Load recent inference data from S3 (last 1, 2, or 3 months)"""
    s3_manager = get_s3_manager()
    return s3_manager.load_inference_data_from_s3(months_recent, local_data_dir)


def get_data_timeline_info() -> Dict[str, str]:
    """Get information about available data timelines"""
    s3_manager = get_s3_manager()
    return s3_manager.get_data_timeline_info()
