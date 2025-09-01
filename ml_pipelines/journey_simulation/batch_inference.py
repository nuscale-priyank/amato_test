#!/usr/bin/env python3
"""
Journey Simulation Batch Inference Pipeline

This module provides batch inference capabilities for customer journey simulation models.
It loads trained models and performs predictions on new customer data to:
1. Predict customer journey stages
2. Predict conversion probabilities
3. Generate insights and recommendations

Author: Data Science Team
Date: 2024
"""

import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.s3_utils import get_s3_manager
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class JourneySimulationBatchInference:
    """
    Batch inference pipeline for customer journey simulation models.
    
    This class handles:
    - Loading trained journey simulation models
    - Preparing inference data
    - Performing predictions
    - Generating reports and visualizations
    """
    
    def __init__(self, config_path=None):
        """Initialize the journey simulation batch inference pipeline."""
        if config_path is None:
            config_path = os.path.join(project_root, 'config', 'database_config.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Model paths
        self.models_path = os.path.join(self.config['ml']['model_storage_path'], 'journey_simulation')
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        
        # Data path
        self.data_path = os.path.join(
            self.config['data_pipelines']['parquet_storage']['base_path'],
            self.config['data_pipelines']['parquet_storage']['unified_customer']
        )
    
    def load_trained_models(self):
        """Load trained journey simulation models and metadata."""
        logger.info("📥 Loading trained journey simulation models...")
        # Pull latest models from S3
        try:
            s3_manager = get_s3_manager()
            s3_manager.download_latest_by_suffix("models/journey_simulation", "models/journey_simulation", [".pkl"])
        except Exception as e:
            logger.warning(f"⚠️  Failed to download latest journey models from S3: {e}")
        try:
            # Load models
            model_files = {
                'journey_stage': 'journey_stage_model.pkl',
                'conversion_prediction': 'conversion_prediction_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = os.path.join(self.models_path, filename)
                if os.path.exists(model_path):
                    import joblib
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"✅ Loaded {model_name.replace('_', ' ').title()} model")
                else:
                    logger.warning(f"⚠️ Model file not found: {model_path}")
            
            # Load scalers (optional)
            scaler_files = {
                'journey_stage': 'journey_stage_scaler.pkl',
                'conversion_prediction': 'conversion_prediction_scaler.pkl'
            }
            
            for scaler_name, filename in scaler_files.items():
                scaler_path = os.path.join(self.models_path, filename)
                if os.path.exists(scaler_path):
                    import joblib
                    self.scalers[scaler_name] = joblib.load(scaler_path)
                else:
                    self.scalers[scaler_name] = None
            
            # Create metadata (since training pipeline doesn't save comprehensive metadata)
            self.metadata = {
                'journey_stage': {
                    'feature_columns': [
                        'rfm_score', 'frequency', 'monetary_value', 'recency_days',
                        'avg_order_value', 'days_since_last_purchase', 'total_sessions',
                        'avg_session_duration', 'engagement_score',
                        'purchase_events', 'add_to_cart_events', 'product_views',
                        'cart_behavior', 'session_intensity', 'conversion_efficiency',
                        'search_intensity', 'product_exploration', 'total_events'
                    ],
                    'model_type': 'RandomForestClassifier',
                    'training_date': datetime.now().isoformat()
                },
                'conversion_prediction': {
                    'feature_columns': [
                        'rfm_score', 'frequency', 'monetary_value', 'recency_days',
                        'avg_order_value', 'days_since_last_purchase', 'total_sessions',
                        'avg_session_duration', 'bounce_rate', 'engagement_score',
                        'purchase_events', 'add_to_cart_events', 'product_views',
                        'cart_behavior', 'session_intensity', 'conversion_efficiency',
                        'search_intensity', 'product_exploration', 'purchase_velocity',
                        'days_since_first_visit', 'avg_product_views', 'cart_add_rate',
                        'purchase_rate'
                    ],
                    'model_type': 'RandomForestRegressor',
                    'training_date': datetime.now().isoformat()
                }
            }
            
            logger.info(f"✅ Loaded {len(self.models)} journey simulation models")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
            return False
    
    def load_inference_data(self):
        """Load data for inference."""
        logger.info("📊 Loading inference data...")
        
        try:
            if not os.path.exists(self.data_path):
                logger.error(f"❌ Data file not found: {self.data_path}")
                return None
            
            df = pd.read_parquet(self.data_path)
            logger.info(f"✅ Loaded inference data: {len(df)} customers")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            return None
    
    def prepare_features(self, df, model_name):
        """Prepare features for inference."""
        logger.info(f"🔧 Preparing features for {model_name}...")
        
        try:
            feature_columns = self.metadata[model_name]['feature_columns']
            
            # Check for missing features
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                logger.warning(f"⚠️ Missing features for {model_name}: {missing_features}")
            
            # Use only available features
            available_features = [col for col in feature_columns if col in df.columns]
            
            # Fill missing values
            df_features = df[['customer_id'] + available_features].copy()
            df_features[available_features] = df_features[available_features].fillna(0)
            
            logger.info(f"✅ Prepared {len(df_features)} customers for {model_name}")
            return df_features
            
        except Exception as e:
            logger.error(f"❌ Error preparing features: {str(e)}")
            return None
    
    def perform_journey_stage_prediction(self, df_features):
        """Perform journey stage prediction."""
        logger.info("🛤️ Performing journey stage prediction...")
        
        feature_columns = self.metadata['journey_stage']['feature_columns']
        available_features = [col for col in feature_columns if col in df_features.columns]
        X = df_features[available_features].values
        
        # Scale features if scaler exists
        if self.scalers['journey_stage'] is not None:
            X_scaled = self.scalers['journey_stage'].transform(X)
        else:
            X_scaled = X  # No scaling if no scaler
        
        # Predict journey stage
        stage_predictions = self.models['journey_stage'].predict(X_scaled)
        
        # Get prediction probabilities
        stage_probabilities = self.models['journey_stage'].predict_proba(X_scaled)
        stage_confidences = np.max(stage_probabilities, axis=1)
        
        # Create results
        results = df_features[['customer_id']].copy()
        results['predicted_journey_stage'] = stage_predictions
        results['stage_confidence'] = stage_confidences
        
        # Add stage categories
        results['stage_category'] = pd.cut(
            results['stage_confidence'],
            bins=[0, 0.5, 0.7, 0.9, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        logger.info(f"✅ Journey stage prediction completed: {len(results)} predictions")
        return results
    
    def perform_conversion_prediction(self, df_features):
        """Perform conversion probability prediction."""
        logger.info("🎯 Performing conversion probability prediction...")
        
        feature_columns = self.metadata['conversion_prediction']['feature_columns']
        available_features = [col for col in feature_columns if col in df_features.columns]
        X = df_features[available_features].values
        
        # Scale features if scaler exists
        if self.scalers['conversion_prediction'] is not None:
            X_scaled = self.scalers['conversion_prediction'].transform(X)
        else:
            X_scaled = X  # No scaling if no scaler
        
        # Predict conversion probability
        conversion_probabilities = self.models['conversion_prediction'].predict(X_scaled)
        
        # Create results
        results = df_features[['customer_id']].copy()
        results['predicted_conversion_probability'] = conversion_probabilities
        results['conversion_confidence'] = 0.85  # Placeholder confidence
        
        # Add conversion categories
        results['conversion_category'] = pd.cut(
            results['predicted_conversion_probability'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Calculate expected value
        results['expected_value'] = results['predicted_conversion_probability'] * 100  # Placeholder calculation
        
        logger.info(f"✅ Conversion prediction completed: {len(results)} predictions")
        return results
    
    def generate_journey_report(self, results, model_name):
        """Generate journey simulation report."""
        logger.info(f"📋 Generating {model_name} report...")
        
        if model_name == 'journey_stage':
            prediction_col = 'predicted_journey_stage'
            confidence_col = 'stage_confidence'
            category_col = 'stage_category'
        else:
            prediction_col = 'predicted_conversion_probability'
            confidence_col = 'conversion_confidence'
            category_col = 'conversion_category'
        
        # Summary statistics
        summary_stats = {
            'total_customers': len(results),
            'avg_confidence': results[confidence_col].mean(),
            'min_confidence': results[confidence_col].min(),
            'max_confidence': results[confidence_col].max()
        }
        
        # Distribution analysis
        if model_name == 'journey_stage':
            stage_dist = results[prediction_col].value_counts()
            summary_stats['journey_stage_distribution'] = stage_dist.to_dict()
            summary_stats['most_common_stage'] = stage_dist.index[0]
            summary_stats['stage_diversity'] = len(stage_dist)
        else:
            avg_conversion_prob = results[prediction_col].mean()
            summary_stats['avg_conversion_probability'] = avg_conversion_prob
            summary_stats['high_conversion_customers'] = (results[prediction_col] > 0.7).sum()
            summary_stats['low_conversion_customers'] = (results[prediction_col] < 0.3).sum()
        
        # Category distribution
        category_dist = results[category_col].value_counts()
        summary_stats['confidence_category_distribution'] = category_dist.to_dict()
        
        # High-confidence predictions
        high_confidence_threshold = results[confidence_col].quantile(0.8)
        high_confidence_customers = results[results[confidence_col] >= high_confidence_threshold]
        
        report = {
            'journey_info': {
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
    
    def save_journey_results(self, results, model_name, output_dir=None):
        """Save journey simulation results."""
        logger.info(f"💾 Saving {model_name} results...")
        
        if output_dir is None:
            output_dir = os.path.join(self.config['ml']['model_storage_path'], 'journey_simulation', 'inference_results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f'{model_name}_results_{timestamp}.parquet')
        results.to_parquet(results_file, index=False)
        
        # Save report
        report = self.generate_journey_report(results, model_name)
        report_file = os.path.join(output_dir, f'{model_name}_report_{timestamp}.yaml')
        with open(report_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f"✅ Results saved to {results_file}")
        logger.info(f"✅ Report saved to {report_file}")
        # Upload to S3 directly
        try:
            s3_manager = get_s3_manager()
            s3_manager.upload_file(results_file, "models/journey_simulation/inference_results")
            s3_manager.upload_file(report_file, "models/journey_simulation/inference_results")
        except Exception as e:
            logger.warning(f"⚠️  Failed to upload journey outputs to S3: {e}")
        
        return results_file, report_file
    
    def create_journey_visualizations(self, results, model_name, output_dir=None):
        """Create journey simulation visualizations."""
        logger.info(f"📊 Creating {model_name} visualizations...")
        
        if output_dir is None:
            output_dir = os.path.join(self.config['ml']['model_storage_path'], 'journey_simulation', 'inference_results')
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if model_name == 'journey_stage':
            # Journey stage distribution
            fig1 = px.pie(
                values=results['predicted_journey_stage'].value_counts().values,
                names=results['predicted_journey_stage'].value_counts().index,
                title='Customer Journey Stage Distribution'
            )
            html1 = os.path.join(output_dir, f'{model_name}_stage_distribution_{timestamp}.html')
            fig1.write_html(html1)
            
            # Stage confidence distribution
            fig2 = px.histogram(
                results,
                x='stage_confidence',
                title='Journey Stage Prediction Confidence Distribution',
                nbins=30
            )
            html2 = os.path.join(output_dir, f'{model_name}_confidence_distribution_{timestamp}.html')
            fig2.write_html(html2)
            
        else:
            # Conversion probability distribution
            fig1 = px.histogram(
                results,
                x='predicted_conversion_probability',
                title='Conversion Probability Distribution',
                nbins=30
            )
            html3 = os.path.join(output_dir, f'{model_name}_conversion_distribution_{timestamp}.html')
            fig1.write_html(html3)
            
            # Conversion category distribution
            fig2 = px.pie(
                values=results['conversion_category'].value_counts().values,
                names=results['conversion_category'].value_counts().index,
                title='Conversion Category Distribution'
            )
            html4 = os.path.join(output_dir, f'{model_name}_conversion_categories_{timestamp}.html')
            fig2.write_html(html4)
        # Upload visualizations to S3
        try:
            s3_manager = get_s3_manager()
            for f in [html1, html2, html3, html4]:
                s3_manager.upload_file(f, "models/journey_simulation/inference_results")
        except Exception as e:
            logger.warning(f"⚠️  Failed to upload journey visualizations to S3: {e}")
        
        logger.info(f"✅ Visualizations saved to {output_dir}")
    
    def run_batch_inference(self, data_path=None, models=None):
        """Run complete batch inference pipeline."""
        logger.info("🚀 Starting Journey Simulation Batch Inference...")
        
        # Override data path if provided
        if data_path is not None:
            self.data_path = data_path
        
        try:
            # Load models
            if not self.load_trained_models():
                raise Exception("Failed to load trained models")
            
            # Load data
            df = self.load_inference_data()
            if df is None:
                raise Exception("Failed to load inference data")
            
            all_results = {}
            
            # Journey Stage Prediction
            logger.info("🛤️ Running Journey Stage Prediction...")
            df_features = self.prepare_features(df, 'journey_stage')
            if df_features is not None:
                results = self.perform_journey_stage_prediction(df_features)
                results_file, report_file = self.save_journey_results(results, 'journey_stage')
                self.create_journey_visualizations(results, 'journey_stage')
                all_results['journey_stage'] = results
                logger.info("✅ journey_stage batch inference completed")
            
            # Conversion Prediction
            logger.info("🎯 Running Conversion Prediction...")
            df_features = self.prepare_features(df, 'conversion_prediction')
            if df_features is not None:
                results = self.perform_conversion_prediction(df_features)
                results_file, report_file = self.save_journey_results(results, 'conversion_prediction')
                self.create_journey_visualizations(results, 'conversion_prediction')
                all_results['conversion_prediction'] = results
                logger.info("✅ conversion_prediction batch inference completed")
            
            logger.info("=" * 60)
            logger.info("🎉 JOURNEY SIMULATION BATCH INFERENCE COMPLETED!")
            logger.info("=" * 60)
            logger.info(f"📊 Processed {len(df)} customers")
            logger.info(f"🎯 Ran inference for {len(all_results)} models")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Error in batch inference: {str(e)}")
            raise


def main():
    """Main function to run journey simulation batch inference."""
    try:
        inference = JourneySimulationBatchInference()
        results = inference.run_batch_inference()
        
        print("\n🎉 Journey Simulation Batch Inference completed successfully!")
        print("📊 Results saved to models/journey_simulation/inference_results/")
        print("📢 Ready for journey optimization!")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
