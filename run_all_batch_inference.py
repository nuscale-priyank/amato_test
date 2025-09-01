#!/usr/bin/env python3
"""
AMATO Production - Master Batch Inference Orchestrator
Runs all batch inference pipelines for comprehensive customer analytics
"""

import logging
import os
from datetime import datetime
import yaml
from utils.s3_utils import get_s3_manager

# Import batch inference classes
from ml_pipelines.customer_segmentation.batch_inference import CustomerSegmentationBatchInference
from ml_pipelines.forecasting.batch_inference import ForecastingBatchInference
from ml_pipelines.journey_simulation.batch_inference import JourneySimulationBatchInference
from ml_pipelines.campaign_optimization.batch_inference import CampaignOptimizationBatchInference

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MasterBatchInferenceOrchestrator:
    def __init__(self, config_path='config/database_config.yaml'):
        self.config = self.load_config(config_path)
        self.results = {}
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def run_customer_segmentation_inference(self, data_path=None):
        """Run customer segmentation batch inference"""
        logger.info("🎯 Running Customer Segmentation Batch Inference...")
        
        try:
            inference = CustomerSegmentationBatchInference()
            results = inference.run_batch_inference(data_path=data_path)
            self.results['customer_segmentation'] = results
            
            logger.info("✅ Customer Segmentation inference completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in customer segmentation inference: {e}")
            raise
    
    def run_forecasting_inference(self, data_path=None):
        """Run forecasting batch inference"""
        logger.info("📈 Running Forecasting Batch Inference...")
        
        try:
            inference = ForecastingBatchInference()
            results = inference.run_batch_inference(data_path=data_path)
            self.results['forecasting'] = results
            
            logger.info("✅ Forecasting inference completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in forecasting inference: {e}")
            raise
    
    def run_journey_simulation_inference(self, data_path=None):
        """Run journey simulation batch inference"""
        logger.info("🛤️ Running Journey Simulation Batch Inference...")
        
        try:
            inference = JourneySimulationBatchInference()
            results = inference.run_batch_inference(data_path=data_path)
            self.results['journey_simulation'] = results
            
            logger.info("✅ Journey Simulation inference completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in journey simulation inference: {e}")
            raise
    
    def run_campaign_optimization_inference(self, data_path=None):
        """Run campaign optimization batch inference"""
        logger.info("📢 Running Campaign Optimization Batch Inference...")
        
        try:
            inference = CampaignOptimizationBatchInference()
            results = inference.run_batch_inference(data_path=data_path)
            self.results['campaign_optimization'] = results
            
            logger.info("✅ Campaign Optimization inference completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in campaign optimization inference: {e}")
            raise
    
    def generate_master_report(self):
        """Generate comprehensive master report"""
        logger.info("📋 Generating master batch inference report...")
        
        # Count total predictions
        total_predictions = 0
        total_models = 0
        pipeline_summary = {}
        
        for pipeline_name, pipeline_results in self.results.items():
            pipeline_predictions = 0
            pipeline_models = len(pipeline_results)
            
            for model_name, model_results in pipeline_results.items():
                if 'results' in model_results:
                    pipeline_predictions += len(model_results['results'])
            
            total_predictions += pipeline_predictions
            total_models += pipeline_models
            
            pipeline_summary[pipeline_name] = {
                'models_run': pipeline_models,
                'total_predictions': pipeline_predictions,
                'models': list(pipeline_results.keys())
            }
        
        # Create master report
        master_report = {
            'master_inference_info': {
                'execution_date': datetime.now().isoformat(),
                'total_pipelines': len(self.results),
                'total_models': total_models,
                'total_predictions': total_predictions,
                'data_source': 'unified_customer_dataset.parquet'
            },
            'pipeline_summary': pipeline_summary,
            'business_insights': {
                'customer_segmentation': {
                    'purpose': 'Customer targeting and personalization',
                    'outputs': ['Customer segments', 'Segment characteristics', 'Targeting recommendations']
                },
                'forecasting': {
                    'purpose': 'Revenue and performance planning',
                    'outputs': ['Revenue predictions', 'CTR forecasts', 'Planning insights']
                },
                'journey_simulation': {
                    'purpose': 'Customer experience optimization',
                    'outputs': ['Journey stage predictions', 'Conversion probabilities', 'UX recommendations']
                },
                'campaign_optimization': {
                    'purpose': 'Marketing efficiency improvement',
                    'outputs': ['Campaign success predictions', 'Budget optimization', 'ROI estimates']
                }
            },
            'next_steps': [
                'Review inference results in respective output directories',
                'Analyze visualizations for business insights',
                'Use predictions for targeted marketing campaigns',
                'Implement recommendations for customer experience improvement',
                'Monitor model performance and retrain as needed'
            ]
        }
        
        # Save master report
        output_dir = 'models/batch_inference_results'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(output_dir, f'master_batch_inference_report_{timestamp}.yaml')
        
        with open(report_file, 'w') as f:
            yaml.dump(master_report, f, default_flow_style=False)
        
        logger.info(f"✅ Master report saved to {report_file}")
        return master_report
    
    def run_all_batch_inference(self, data_path=None):
        """Run all batch inference pipelines"""
        logger.info("🚀 Starting Master Batch Inference Orchestrator...")
        logger.info("=" * 80)
        
        try:
            # Pull latest models and unified dataset from S3
            try:
                s3_manager = get_s3_manager()
                s3_manager.load_models_from_s3("models")
                s3_manager.load_data_from_s3("data_pipelines/unified_dataset/output")
            except Exception as sync_err:
                logger.warning(f"⚠️  Failed to load from S3, proceeding with local artifacts if present: {sync_err}")
            # Run all inference pipelines
            self.run_customer_segmentation_inference(data_path)
            self.run_forecasting_inference(data_path)
            self.run_journey_simulation_inference(data_path)
            self.run_campaign_optimization_inference(data_path)
            
            # Generate master report
            master_report = self.generate_master_report()
            
            logger.info("=" * 80)
            logger.info("🎉 MASTER BATCH INFERENCE COMPLETED!")
            logger.info("=" * 80)
            logger.info(f"📊 Total pipelines: {len(self.results)}")
            logger.info(f"🎯 Total models: {sum(len(pipeline) for pipeline in self.results.values())}")
            logger.info(f"📈 Total predictions: {master_report['master_inference_info']['total_predictions']}")
            logger.info(f"📁 Results saved to: models/*/inference_results/")
            logger.info(f"📋 Master report: models/batch_inference_results/")
            
            # Sync outputs to S3
            try:
                s3_manager.sync_inference_results_to_s3("models")
            except Exception as out_sync_err:
                logger.warning(f"⚠️  Failed to sync batch inference results to S3: {out_sync_err}")
            
            return self.results, master_report
            
        except Exception as e:
            logger.error(f"❌ Error in master batch inference: {e}")
            raise

def main():
    """Main function"""
    try:
        orchestrator = MasterBatchInferenceOrchestrator()
        results, master_report = orchestrator.run_all_batch_inference()
        
        print(f"\n🎉 Master Batch Inference completed successfully!")
        print(f"📊 Processed all 4 ML pipelines")
        print(f"🎯 Generated comprehensive customer analytics")
        print(f"📁 Results available in models/*/inference_results/")
        print(f"📋 Master report: models/batch_inference_results/")
        print(f"🚀 Ready for business decision making!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
