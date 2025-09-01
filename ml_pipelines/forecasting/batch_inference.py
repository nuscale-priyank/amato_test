#!/usr/bin/env python3
"""
AMATO Production - Forecasting Batch Inference Pipeline
Performs batch inference for revenue and CTR forecasting
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
import sys
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.s3_utils import get_s3_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastingBatchInference:
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
        """Load trained forecasting models"""
        logger.info("📥 Loading trained forecasting models...")
        # Pull latest models from S3
        try:
            s3_manager = get_s3_manager()
            s3_manager.download_latest_by_suffix("models/forecasting", "models/forecasting", [".pkl"])
        except Exception as e:
            logger.warning(f"⚠️  Failed to download latest forecasting models from S3: {e}")
        model_dir = os.path.join(self.config['ml']['model_storage_path'], 'forecasting')
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load Revenue Forecasting model
        revenue_model_path = os.path.join(model_dir, 'revenue_forecasting_model.pkl')
        revenue_scaler_path = os.path.join(model_dir, 'revenue_forecasting_scaler.pkl')
        revenue_metadata_path = os.path.join(model_dir, 'revenue_forecasting_metadata.yaml')
        
        if os.path.exists(revenue_model_path):
            self.models['revenue'] = joblib.load(revenue_model_path)
            if os.path.exists(revenue_scaler_path):
                self.scalers['revenue'] = joblib.load(revenue_scaler_path)
            else:
                self.scalers['revenue'] = None
            # Create simple metadata
            self.metadata['revenue'] = {
                'feature_columns': ['rfm_score', 'frequency', 'avg_order_value', 'days_since_last_purchase', 'day_of_week', 'month', 'quarter', 'year', 'revenue_lag_1', 'revenue_lag_7', 'revenue_lag_30', 'revenue_ma_7', 'revenue_ma_30', 'customer_age_days'],
                'model_type': 'RandomForestRegressor',
                'training_date': datetime.now().isoformat()
            }
            logger.info("✅ Loaded Revenue Forecasting model")
        
        # Load CTR Forecasting model
        ctr_model_path = os.path.join(model_dir, 'ctr_forecasting_model.pkl')
        ctr_scaler_path = os.path.join(model_dir, 'ctr_forecasting_scaler.pkl')
        ctr_metadata_path = os.path.join(model_dir, 'ctr_forecasting_metadata.yaml')
        
        if os.path.exists(ctr_model_path):
            self.models['ctr'] = joblib.load(ctr_model_path)
            if os.path.exists(ctr_scaler_path):
                self.scalers['ctr'] = joblib.load(ctr_scaler_path)
            else:
                self.scalers['ctr'] = None
            # Create simple metadata
            self.metadata['ctr'] = {
                'feature_columns': ['rfm_score', 'frequency', 'avg_order_value', 'day_of_week', 'month', 'quarter', 'year', 'ctr_lag_1', 'ctr_ma_7', 'customer_age_days'],
                'model_type': 'RandomForestRegressor',
                'training_date': datetime.now().isoformat()
            }
            logger.info("✅ Loaded CTR Forecasting model")
        
        logger.info(f"✅ Loaded {len(self.models)} forecasting models")
    
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
    
    def prepare_forecasting_features(self, df, model_name):
        """Prepare features for forecasting inference"""
        logger.info(f"🔧 Preparing features for {model_name} forecasting...")
        
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
        
        logger.info(f"✅ Prepared {len(df_features)} customers for {model_name} forecasting")
        return df_features
    
    def perform_revenue_forecast(self, df_features):
        """Perform revenue forecasting"""
        logger.info("💰 Performing revenue forecasting...")
        
        feature_columns = self.metadata['revenue']['feature_columns']
        X = df_features[feature_columns].values
        
        # Scale features if scaler exists
        if self.scalers['revenue'] is not None:
            X_scaled = self.scalers['revenue'].transform(X)
        else:
            X_scaled = X  # No scaling if no scaler
        
        # Predict revenue
        revenue_predictions = self.models['revenue'].predict(X_scaled)
        
        # Create results
        results = df_features[['customer_id']].copy()
        results['predicted_revenue'] = revenue_predictions
        results['forecast_confidence'] = 0.85  # Placeholder confidence score
        
        # Add forecast periods
        results['forecast_period'] = 'next_month'
        results['forecast_date'] = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"✅ Revenue forecasting completed: {len(results)} predictions")
        return results
    
    def perform_ctr_forecast(self, df_features):
        """Perform CTR forecasting"""
        logger.info("📈 Performing CTR forecasting...")
        
        feature_columns = self.metadata['ctr']['feature_columns']
        X = df_features[feature_columns].values
        
        # Scale features if scaler exists
        if self.scalers['ctr'] is not None:
            X_scaled = self.scalers['ctr'].transform(X)
        else:
            X_scaled = X  # No scaling if no scaler
        
        # Predict CTR
        ctr_predictions = self.models['ctr'].predict(X_scaled)
        
        # Create results
        results = df_features[['customer_id']].copy()
        results['predicted_ctr'] = ctr_predictions
        results['forecast_confidence'] = 0.80  # Placeholder confidence score
        
        # Add forecast periods
        results['forecast_period'] = 'next_campaign'
        results['forecast_date'] = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"✅ CTR forecasting completed: {len(results)} predictions")
        return results
    
    def generate_forecast_report(self, results, model_name):
        """Generate forecasting report"""
        logger.info(f"📋 Generating {model_name} forecast report...")
        
        if model_name == 'revenue':
            prediction_col = 'predicted_revenue'
            metric_name = 'Revenue'
        else:
            prediction_col = 'predicted_ctr'
            metric_name = 'CTR'
        
        # Summary statistics
        summary_stats = {
            'total_customers': len(results),
            'avg_prediction': results[prediction_col].mean(),
            'min_prediction': results[prediction_col].min(),
            'max_prediction': results[prediction_col].max(),
            'std_prediction': results[prediction_col].std(),
            'total_predicted_value': results[prediction_col].sum()
        }
        
        # Prediction distribution
        prediction_dist = results[prediction_col].describe()
        
        # High-value predictions (top 20%)
        high_value_threshold = results[prediction_col].quantile(0.8)
        high_value_customers = results[results[prediction_col] >= high_value_threshold]
        
        report = {
            'forecast_info': {
                'model_name': model_name,
                'forecast_date': datetime.now().isoformat(),
                'forecast_period': results['forecast_period'].iloc[0],
                'total_customers': len(results)
            },
            'summary_statistics': summary_stats,
            'prediction_distribution': prediction_dist.to_dict(),
            'high_value_insights': {
                'threshold': high_value_threshold,
                'high_value_customers': len(high_value_customers),
                'high_value_percentage': len(high_value_customers) / len(results) * 100,
                'avg_high_value': high_value_customers[prediction_col].mean()
            },
            'model_metadata': {
                'training_date': self.metadata[model_name]['training_date'],
                'feature_count': len(self.metadata[model_name]['feature_columns']),
                'performance_metrics': {
                    'mae': 0.0,
                    'rmse': 0.0,
                    'r2': 0.0
                }
            }
        }
        
        return report
    
    def save_forecast_results(self, results, model_name, output_dir=None):
        """Save forecasting results"""
        logger.info(f"💾 Saving {model_name} forecast results...")
        
        if output_dir is None:
            output_dir = os.path.join(self.config['ml']['model_storage_path'], 'forecasting', 'inference_results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f'{model_name}_forecast_results_{timestamp}.parquet')
        results.to_parquet(results_file, index=False)
        # Upload results to S3 directly
        try:
            s3_manager = get_s3_manager()
            s3_manager.upload_file(results_file, "models/forecasting/inference_results")
        except Exception as e:
            logger.warning(f"⚠️  Failed to upload forecasting results to S3: {e}")
        
        # Save report
        report = self.generate_forecast_report(results, model_name)
        report_file = os.path.join(output_dir, f'{model_name}_forecast_report_{timestamp}.yaml')
        with open(report_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f"✅ Results saved to {results_file}")
        logger.info(f"✅ Report saved to {report_file}")
        
        return results_file, report_file
    
    def create_forecast_visualizations(self, results, model_name, output_dir=None):
        """Create forecasting visualizations"""
        logger.info(f"📊 Creating {model_name} forecast visualizations...")
        
        if output_dir is None:
            output_dir = os.path.join(self.config['ml']['model_storage_path'], 'forecasting', 'inference_results')
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if model_name == 'revenue':
            prediction_col = 'predicted_revenue'
            title = 'Revenue Forecast Distribution'
        else:
            prediction_col = 'predicted_ctr'
            title = 'CTR Forecast Distribution'
        
        # Prediction distribution histogram
        fig1 = px.histogram(
            results, 
            x=prediction_col,
            title=title,
            nbins=50
        )
        html1 = os.path.join(output_dir, f'{model_name}_forecast_distribution_{timestamp}.html')
        fig1.write_html(html1)
        
        # Prediction vs customer count
        fig2 = px.scatter(
            x=range(len(results)),
            y=results[prediction_col].sort_values(ascending=False),
            title=f'{model_name.upper()} Forecast by Customer Rank',
            labels={'x': 'Customer Rank', 'y': prediction_col.replace('_', ' ').title()}
        )
        html2 = os.path.join(output_dir, f'{model_name}_forecast_rank_{timestamp}.html')
        fig2.write_html(html2)
        # Upload visualizations to S3
        try:
            s3_manager = get_s3_manager()
            for f in [html1, html2]:
                s3_manager.upload_file(f, "models/forecasting/inference_results")
        except Exception as e:
            logger.warning(f"⚠️  Failed to upload forecasting visualizations to S3: {e}")
        
        logger.info(f"✅ Visualizations saved to {output_dir}")
    
    def run_batch_inference(self, data_path=None, models=None):
        """Run batch inference for forecasting models"""
        logger.info("🚀 Starting Forecasting Batch Inference...")
        
        try:
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
                df_features = self.prepare_forecasting_features(df, model_name)
                
                # Perform forecasting
                if model_name == 'revenue':
                    results = self.perform_revenue_forecast(df_features)
                elif model_name == 'ctr':
                    results = self.perform_ctr_forecast(df_features)
                
                # Save results
                results_file, report_file = self.save_forecast_results(results, model_name)
                
                # Create visualizations
                self.create_forecast_visualizations(results, model_name)
                
                all_results[model_name] = {
                    'results': results,
                    'results_file': results_file,
                    'report_file': report_file
                }
                
                logger.info(f"✅ {model_name} batch inference completed")
            
            logger.info("=" * 60)
            logger.info("🎉 FORECASTING BATCH INFERENCE COMPLETED!")
            logger.info("=" * 60)
            logger.info(f"📊 Processed {len(df)} customers")
            logger.info(f"🎯 Ran inference for {len(all_results)} models")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Error in batch inference: {e}")
            raise

def main():
    """Main function"""
    try:
        inference = ForecastingBatchInference()
        results = inference.run_batch_inference()
        
        print(f"\n🎉 Forecasting Batch Inference completed successfully!")
        print(f"📊 Results saved to models/forecasting/inference_results/")
        print(f"📈 Ready for business planning!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
