#!/usr/bin/env python3
"""
AMATO Production - Campaign Optimization Batch Inference Pipeline
Performs batch inference for campaign success prediction and budget optimization
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from utils.s3_utils import get_s3_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CampaignOptimizationBatchInference:
    def __init__(self, config_path='config/database_config.yaml'):
        self.config = self.load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_trained_models(self):
        """Load trained campaign optimization models"""
        logger.info("📥 Loading trained campaign optimization models...")
        
        model_dir = os.path.join(self.config['ml']['model_storage_path'], 'campaign_optimization')
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load Campaign Success model
        success_model_path = os.path.join(model_dir, 'campaign_success_model.pkl')
        success_scaler_path = os.path.join(model_dir, 'campaign_success_scaler.pkl')
        success_metadata_path = os.path.join(model_dir, 'campaign_success_metadata.yaml')
        
        if os.path.exists(success_model_path):
            self.models['campaign_success'] = joblib.load(success_model_path)
            if os.path.exists(success_scaler_path):
                self.scalers['campaign_success'] = joblib.load(success_scaler_path)
            else:
                self.scalers['campaign_success'] = None
            # Create simple metadata
            self.metadata['campaign_success'] = {
                'feature_columns': ['rfm_score', 'frequency', 'avg_order_value', 'days_since_last_purchase', 'campaign_count', 'total_campaign_revenue', 'avg_roas', 'avg_ctr', 'campaign_roas', 'campaign_cpa', 'campaign_ctr', 'campaign_cvr', 'customer_campaign_affinity', 'target_audience_match', 'channel_effectiveness', 'campaign_month', 'campaign_quarter', 'campaign_day_of_week'],
                'model_type': 'RandomForestClassifier',
                'training_date': datetime.now().isoformat()
            }
            logger.info("✅ Loaded Campaign Success model")
        
        # Load Budget Optimization model
        budget_model_path = os.path.join(model_dir, 'budget_optimization_model.pkl')
        budget_scaler_path = os.path.join(model_dir, 'budget_optimization_scaler.pkl')
        budget_metadata_path = os.path.join(model_dir, 'budget_optimization_metadata.yaml')
        
        if os.path.exists(budget_model_path):
            self.models['budget_optimization'] = joblib.load(budget_model_path)
            if os.path.exists(budget_scaler_path):
                self.scalers['budget_optimization'] = joblib.load(budget_scaler_path)
            else:
                self.scalers['budget_optimization'] = None
            # Create simple metadata
            self.metadata['budget_optimization'] = {
                'feature_columns': ['rfm_score', 'frequency', 'avg_order_value', 'days_since_last_purchase', 'campaign_count', 'total_campaign_revenue', 'avg_roas', 'avg_ctr', 'campaign_roas', 'campaign_cpa', 'campaign_ctr', 'campaign_cvr', 'customer_campaign_affinity', 'target_audience_match', 'channel_effectiveness', 'campaign_month', 'campaign_quarter', 'campaign_day_of_week'],
                'model_type': 'RandomForestRegressor',
                'training_date': datetime.now().isoformat()
            }
            logger.info("✅ Loaded Budget Optimization model")
        
        logger.info(f"✅ Loaded {len(self.models)} campaign optimization models")
    
    def load_inference_data(self, data_path=None):
        """Load data for batch inference"""
        logger.info("📊 Loading inference data...")
        
        if data_path is None:
            data_path = 'data_pipelines/unified_dataset/output/unified_customer_dataset.parquet'
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Inference data not found: {data_path}")
        
        df = pd.read_parquet(data_path)
        logger.info(f"✅ Loaded inference data: {len(df)} customers")
        return df
    
    def prepare_campaign_features(self, df, model_name):
        """Prepare features for campaign optimization"""
        logger.info(f"🔧 Preparing features for {model_name}...")
        
        # Get feature columns from metadata
        feature_columns = self.metadata[model_name]['feature_columns']
        
        # Select available features
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = [col for col in feature_columns if col not in df.columns]
        
        if missing_features:
            logger.warning(f"⚠️ Missing features for {model_name}: {missing_features}")
            for feature in missing_features:
                df[feature] = 0
        
        # Select features and handle missing values
        df_features = df[['customer_id'] + feature_columns].copy()
        df_features = df_features.fillna(0)
        
        logger.info(f"✅ Prepared {len(df_features)} customers for {model_name}")
        return df_features
    
    def perform_campaign_success_prediction(self, df_features):
        """Perform campaign success prediction"""
        logger.info("🎯 Performing campaign success prediction...")
        
        feature_columns = self.metadata['campaign_success']['feature_columns']
        X = df_features[feature_columns].values
        
        # Scale features if scaler exists
        if self.scalers['campaign_success'] is not None:
            X_scaled = self.scalers['campaign_success'].transform(X)
        else:
            X_scaled = X  # No scaling if no scaler
        
        # Predict campaign success
        success_predictions = self.models['campaign_success'].predict(X_scaled)
        
        # Get prediction probabilities
        success_probabilities = self.models['campaign_success'].predict_proba(X_scaled)
        # Get success probabilities
        if success_probabilities.shape[1] > 1:
            success_probs = success_probabilities[:, 1]  # Probability of success
        else:
            success_probs = success_probabilities[:, 0]  # Only one class predicted
        
        # Create results
        results = df_features[['customer_id']].copy()
        results['predicted_campaign_success'] = success_predictions  # Keep as string labels
        results['success_probability'] = success_probs
        results['success_confidence'] = np.maximum(success_probs, 1 - success_probs)
        
        # Add success categories
        results['success_category'] = pd.cut(
            results['success_probability'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        logger.info(f"✅ Campaign success prediction completed: {len(results)} predictions")
        return results
    
    def perform_budget_optimization(self, df_features):
        """Perform budget optimization prediction"""
        logger.info("💰 Performing budget optimization prediction...")
        
        feature_columns = self.metadata['budget_optimization']['feature_columns']
        X = df_features[feature_columns].values
        
        # Scale features if scaler exists
        if self.scalers['budget_optimization'] is not None:
            X_scaled = self.scalers['budget_optimization'].transform(X)
        else:
            X_scaled = X  # No scaling if no scaler
        
        # Predict optimal budget
        budget_predictions = self.models['budget_optimization'].predict(X_scaled)
        
        # Create results
        results = df_features[['customer_id']].copy()
        results['predicted_optimal_budget'] = budget_predictions.astype(float)  # Ensure numeric
        results['budget_confidence'] = 0.85  # Placeholder confidence
        
        # Add budget categories
        results['budget_category'] = pd.cut(
            results['predicted_optimal_budget'],
            bins=[0, 100, 500, 1000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Premium']
        )
        
        # Calculate ROI estimates
        results['estimated_roi'] = results['predicted_optimal_budget'] * 0.15  # Placeholder ROI calculation
        results['roi_category'] = pd.cut(
            results['estimated_roi'],
            bins=[0, 50, 100, 200, float('inf')],
            labels=['Low', 'Medium', 'High', 'Premium']
        )
        
        logger.info(f"✅ Budget optimization completed: {len(results)} predictions")
        return results
    
    def generate_campaign_report(self, results, model_name):
        """Generate campaign optimization report"""
        logger.info(f"📋 Generating {model_name} report...")
        
        if model_name == 'campaign_success':
            prediction_col = 'predicted_campaign_success'
            probability_col = 'success_probability'
            category_col = 'success_category'
            confidence_col = 'success_confidence'
        else:
            prediction_col = 'predicted_optimal_budget'
            probability_col = 'estimated_roi'
            category_col = 'budget_category'
            confidence_col = 'budget_confidence'
        
        # Summary statistics
        summary_stats = {
            'total_customers': len(results),
            'avg_confidence': results[confidence_col].mean(),
            'min_confidence': results[confidence_col].min(),
            'max_confidence': results[confidence_col].max()
        }
        
        # Distribution analysis
        if model_name == 'campaign_success':
            # For string predictions, calculate success rate based on 'high' predictions
            success_rate = (results[prediction_col] == 'high').mean() * 100
            summary_stats['predicted_success_rate'] = success_rate
            summary_stats['avg_success_probability'] = results[probability_col].mean()
            
            # Distribution of predicted success levels
            success_dist = results[prediction_col].value_counts()
            summary_stats['predicted_success_distribution'] = success_dist.to_dict()
            
            category_dist = results[category_col].value_counts()
            summary_stats['success_category_distribution'] = category_dist.to_dict()
        else:
            total_budget = results[prediction_col].sum()
            avg_budget = results[prediction_col].mean()
            summary_stats['total_predicted_budget'] = total_budget
            summary_stats['avg_predicted_budget'] = avg_budget
            summary_stats['total_estimated_roi'] = results[probability_col].sum()
            
            category_dist = results[category_col].value_counts()
            summary_stats['budget_category_distribution'] = category_dist.to_dict()
        
        # High-confidence predictions
        high_confidence_threshold = results[confidence_col].quantile(0.8)
        high_confidence_customers = results[results[confidence_col] >= high_confidence_threshold]
        
        report = {
            'campaign_info': {
                'model_name': model_name,
                'inference_date': datetime.now().isoformat(),
                'total_customers': len(results)
            },
            'summary_statistics': summary_stats,
            'high_confidence_insights': {
                'threshold': high_confidence_threshold,
                'high_confidence_customers': len(high_confidence_customers),
                'high_confidence_percentage': len(high_confidence_customers) / len(results) * 100
            },
            'model_metadata': {
                'training_date': self.metadata[model_name]['training_date'],
                'feature_count': len(self.metadata[model_name]['feature_columns']),
                'performance_metrics': self.metadata[model_name].get('performance_metrics', {
                    'accuracy': 'N/A',
                    'precision': 'N/A',
                    'recall': 'N/A',
                    'f1_score': 'N/A'
                })
            }
        }
        
        return report
    
    def save_campaign_results(self, results, model_name, output_dir=None):
        """Save campaign optimization results"""
        logger.info(f"💾 Saving {model_name} results...")
        
        if output_dir is None:
            output_dir = os.path.join(self.config['ml']['model_storage_path'], 'campaign_optimization', 'inference_results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f'{model_name}_results_{timestamp}.parquet')
        results.to_parquet(results_file, index=False)
        
        # Save report
        report = self.generate_campaign_report(results, model_name)
        report_file = os.path.join(output_dir, f'{model_name}_report_{timestamp}.yaml')
        with open(report_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f"✅ Results saved to {results_file}")
        logger.info(f"✅ Report saved to {report_file}")
        
        return results_file, report_file
    
    def create_campaign_visualizations(self, results, model_name, output_dir=None):
        """Create campaign optimization visualizations"""
        logger.info(f"📊 Creating {model_name} visualizations...")
        
        if output_dir is None:
            output_dir = os.path.join(self.config['ml']['model_storage_path'], 'campaign_optimization', 'inference_results')
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if model_name == 'campaign_success':
            # Success probability distribution
            fig1 = px.histogram(
                results, 
                x='success_probability',
                title='Campaign Success Probability Distribution',
                nbins=30
            )
            fig1.write_html(os.path.join(output_dir, f'{model_name}_success_distribution_{timestamp}.html'))
            
            # Success category distribution
            fig2 = px.pie(
                values=results['success_category'].value_counts().values,
                names=results['success_category'].value_counts().index,
                title='Campaign Success Category Distribution'
            )
            fig2.write_html(os.path.join(output_dir, f'{model_name}_success_categories_{timestamp}.html'))
            
        else:
            # Budget distribution
            fig1 = px.histogram(
                results, 
                x='predicted_optimal_budget',
                title='Predicted Optimal Budget Distribution',
                nbins=30
            )
            fig1.write_html(os.path.join(output_dir, f'{model_name}_budget_distribution_{timestamp}.html'))
            
            # Budget vs ROI scatter
            fig2 = px.scatter(
                results,
                x='predicted_optimal_budget',
                y='estimated_roi',
                color='budget_category',
                title='Budget vs Estimated ROI',
                labels={'predicted_optimal_budget': 'Predicted Budget', 'estimated_roi': 'Estimated ROI'}
            )
            fig2.write_html(os.path.join(output_dir, f'{model_name}_budget_vs_roi_{timestamp}.html'))
        
        logger.info(f"✅ Visualizations saved to {output_dir}")
    
    def run_batch_inference(self, data_path=None, models=None):
        """Run batch inference for campaign optimization models"""
        logger.info("🚀 Starting Campaign Optimization Batch Inference...")
        
        try:
            # Ensure latest models are available locally from S3 (no sync, pull latest files)
            try:
                s3_manager = get_s3_manager()
                s3_manager.download_latest_by_suffix("models/campaign_optimization", "models/campaign_optimization", [".pkl"])
            except Exception as sync_err:
                logger.warning(f"⚠️  Failed to load latest models from S3, proceeding with local models if present: {sync_err}")
            # Load models
            self.load_trained_models()
            
            # Load data
            df = self.load_inference_data(data_path)
            
            # Determine which models to run
            if models is None:
                models = list(self.models.keys())
            
            all_results = {}
            
            for model_name in models:
                if model_name not in self.models:
                    logger.warning(f"⚠️ Model {model_name} not found, skipping...")
                    continue
                
                # Prepare features
                df_features = self.prepare_campaign_features(df, model_name)
                
                # Perform inference
                if model_name == 'campaign_success':
                    results = self.perform_campaign_success_prediction(df_features)
                elif model_name == 'budget_optimization':
                    results = self.perform_budget_optimization(df_features)
                
                # Save results
                results_file, report_file = self.save_campaign_results(results, model_name)
                
                # Create visualizations
                self.create_campaign_visualizations(results, model_name)
                
                all_results[model_name] = {
                    'results': results,
                    'results_file': results_file,
                    'report_file': report_file
                }
                
                logger.info(f"✅ {model_name} batch inference completed")
            
            logger.info("=" * 60)
            logger.info("🎉 CAMPAIGN OPTIMIZATION BATCH INFERENCE COMPLETED!")
            logger.info("=" * 60)
            logger.info(f"📊 Processed {len(df)} customers")
            logger.info(f"🎯 Ran inference for {len(all_results)} models")
            
            # Upload results produced in this run to S3 directly
            try:
                # Upload parquet and yaml plus HTMLs generated in output_dir
                # Since save_campaign_results and create_campaign_visualizations already write files locally,
                # upload the newest files from the output directory
                output_dir = os.path.join(self.config['ml']['model_storage_path'], 'campaign_optimization', 'inference_results')
                for f in os.listdir(output_dir):
                    full = os.path.join(output_dir, f)
                    if os.path.isfile(full) and any(f.endswith(ext) for ext in ['.parquet', '.yaml', '.html']):
                        s3_manager.upload_file(full, "models/campaign_optimization/inference_results")
            except Exception as out_sync_err:
                logger.warning(f"⚠️  Failed to upload campaign inference results to S3: {out_sync_err}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Error in batch inference: {e}")
            raise

def main():
    """Main function"""
    try:
        inference = CampaignOptimizationBatchInference()
        results = inference.run_batch_inference()
        
        print(f"\n🎉 Campaign Optimization Batch Inference completed successfully!")
        print(f"📊 Results saved to models/campaign_optimization/inference_results/")
        print(f"📢 Ready for campaign optimization!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
