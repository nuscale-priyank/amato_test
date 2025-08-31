#!/usr/bin/env python3
"""
Master Script to Train All ML Pipelines
Trains all models for Customer Segmentation, Forecasting, Journey Simulation, and Campaign Optimization
"""

import logging
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_all_pipelines():
    """Train all ML pipelines"""
    logger.info("🚀 Starting Training for All ML Pipelines...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # List of pipeline training scripts
    pipelines = [
        {
            'name': 'Customer Segmentation',
            'script': 'ml_pipelines/customer_segmentation/train_segmentation_models.py',
            'description': 'Customer RFM and behavioral segmentation models'
        },
        {
            'name': 'Forecasting',
            'script': 'ml_pipelines/forecasting/train_forecasting_models.py',
            'description': 'Revenue and CTR forecasting models'
        },
        {
            'name': 'Journey Simulation',
            'script': 'ml_pipelines/journey_simulation/train_journey_models.py',
            'description': 'Customer journey stage and conversion prediction models'
        },
        {
            'name': 'Campaign Optimization',
            'script': 'ml_pipelines/campaign_optimization/train_campaign_models.py',
            'description': 'Campaign success and budget optimization models'
        }
    ]
    
    results = {}
    
    for pipeline in pipelines:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {pipeline['name']} Pipeline")
        logger.info(f"Description: {pipeline['description']}")
        logger.info(f"{'='*60}")
        
        try:
            # Import and run the pipeline
            script_path = pipeline['script']
            if os.path.exists(script_path):
                # Execute the script
                exec(open(script_path).read())
                results[pipeline['name']] = '✅ Success'
                logger.info(f"✅ {pipeline['name']} pipeline completed successfully!")
            else:
                logger.error(f"❌ Script not found: {script_path}")
                results[pipeline['name']] = '❌ Script not found'
                
        except Exception as e:
            logger.error(f"❌ Error training {pipeline['name']} pipeline: {e}")
            results[pipeline['name']] = f'❌ Error: {str(e)}'
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("🎯 ML Pipeline Training Summary")
    logger.info(f"{'='*60}")
    
    for pipeline_name, result in results.items():
        logger.info(f"{pipeline_name}: {result}")
    
    # Count successes
    success_count = sum(1 for result in results.values() if '✅' in result)
    total_count = len(results)
    
    logger.info(f"\nOverall Results: {success_count}/{total_count} pipelines successful")
    
    if success_count == total_count:
        logger.info("🎉 All ML pipelines trained successfully!")
        return True
    else:
        logger.warning(f"⚠️  {total_count - success_count} pipeline(s) failed")
        return False

if __name__ == "__main__":
    success = train_all_pipelines()
    sys.exit(0 if success else 1)
