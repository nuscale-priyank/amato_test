#!/usr/bin/env python3
"""
Script to sync AMATO project files to S3
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.s3_utils import sync_all_to_s3, get_s3_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main function to sync files to S3"""
    logger.info("Starting AMATO S3 sync...")
    
    try:
        # Get S3 manager
        s3_manager = get_s3_manager()
        logger.info(f"Using S3 bucket: {s3_manager.bucket}")
        logger.info(f"Using S3 base path: {s3_manager.base_path}")
        
        # Sync all files
        results = sync_all_to_s3()
        
        # Print results
        for category, file_results in results.items():
            success_count = sum(1 for success in file_results.values() if success)
            total_count = len(file_results)
            logger.info(f"{category}: {success_count}/{total_count} files synced successfully")
            
            if total_count > 0:
                success_rate = (success_count / total_count) * 100
                logger.info(f"{category} success rate: {success_rate:.1f}%")
        
        logger.info("S3 sync completed!")
        
    except Exception as e:
        logger.error(f"Error during S3 sync: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
