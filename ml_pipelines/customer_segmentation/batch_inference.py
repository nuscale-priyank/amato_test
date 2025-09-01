#!/usr/bin/env python3
"""
AMATO Production - Customer Segmentation Batch Inference Pipeline
Performs batch inference on customer data using trained segmentation models
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
from plotly.subplots import make_subplots

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerSegmentationBatchInference:
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
        """Load trained models and metadata"""
        logger.info("📥 Loading trained segmentation models...")
        
        model_dir = os.path.join(self.config['ml']['model_storage_path'], 'customer_segmentation')
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load K-means model
        kmeans_model_path = os.path.join(model_dir, 'kmeans_model.pkl')
        kmeans_scaler_path = os.path.join(model_dir, 'kmeans_scaler.pkl')
        kmeans_metadata_path = os.path.join(model_dir, 'kmeans_metadata.yaml')
        
        if os.path.exists(kmeans_model_path):
            self.models['kmeans'] = joblib.load(kmeans_model_path)
            self.scalers['kmeans'] = joblib.load(kmeans_scaler_path)
            # Load feature columns from CSV instead of YAML
            feature_df = pd.read_csv(os.path.join(model_dir, 'feature_characteristics.csv'))
            self.metadata['kmeans'] = {
                'feature_columns': feature_df['feature'].tolist(),
                'model_type': 'KMeans',
                'training_date': datetime.now().isoformat()
            }
            logger.info("✅ Loaded K-means model")
        
        # Load HDBSCAN model
        hdbscan_model_path = os.path.join(model_dir, 'hdbscan_model.pkl')
        hdbscan_scaler_path = os.path.join(model_dir, 'hdbscan_scaler.pkl')
        hdbscan_metadata_path = os.path.join(model_dir, 'hdbscan_metadata.yaml')
        
        if os.path.exists(hdbscan_model_path):
            self.models['hdbscan'] = joblib.load(hdbscan_model_path)
            self.scalers['hdbscan'] = joblib.load(hdbscan_scaler_path)
            # Use same feature columns as K-means
            self.metadata['hdbscan'] = {
                'feature_columns': self.metadata['kmeans']['feature_columns'],
                'model_type': 'HDBSCAN',
                'training_date': datetime.now().isoformat()
            }
            logger.info("✅ Loaded HDBSCAN model")
        
        logger.info(f"✅ Loaded {len(self.models)} models")
    
    def load_inference_data(self, data_path=None):
        """Load data for batch inference"""
        logger.info("📊 Loading inference data...")
        
        if data_path is None:
            # Use unified dataset
            data_path = 'data_pipelines/unified_dataset/output/unified_customer_dataset.parquet'
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Inference data not found: {data_path}")
        
        df = pd.read_parquet(data_path)
        logger.info(f"✅ Loaded inference data: {len(df)} customers")
        return df
    
    def prepare_inference_features(self, df, model_name):
        """Prepare features for inference"""
        logger.info(f"🔧 Preparing features for {model_name} inference...")
        
        # Get feature columns from metadata
        feature_columns = self.metadata[model_name]['feature_columns']
        
        # Select available features
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = [col for col in feature_columns if col not in df.columns]
        
        if missing_features:
            logger.warning(f"⚠️ Missing features for {model_name}: {missing_features}")
            # Fill missing features with 0
            for feature in missing_features:
                df[feature] = 0
        
        # Select features and handle missing values
        df_features = df[['customer_id'] + feature_columns].copy()
        df_features = df_features.fillna(0)
        
        # Remove customers with no purchase history (same as training)
        df_features = df_features[df_features['monetary_value'] > 0]
        
        logger.info(f"✅ Prepared {len(df_features)} customers for {model_name} inference")
        return df_features
    
    def perform_inference(self, df_features, model_name):
        """Perform inference using trained model"""
        logger.info(f"🎯 Performing {model_name} inference...")
        
        # Prepare features
        feature_columns = self.metadata[model_name]['feature_columns']
        X = df_features[feature_columns].values
        
        # Scale features
        X_scaled = self.scalers[model_name].transform(X)
        
        # Predict segments
        if model_name == 'kmeans':
            segments = self.models[model_name].predict(X_scaled)
        elif model_name == 'hdbscan':
            segments = self.models[model_name].fit_predict(X_scaled)
        
        # Create results dataframe
        results = df_features[['customer_id']].copy()
        results[f'{model_name}_segment'] = segments
        results[f'{model_name}_confidence'] = 1.0  # Placeholder for confidence scores
        
        # Add segment characteristics
        results = self.add_segment_characteristics(results, model_name, df_features)
        
        logger.info(f"✅ {model_name} inference completed: {len(results)} predictions")
        return results
    
    def add_segment_characteristics(self, results, model_name, df_features):
        """Add segment characteristics to results"""
        logger.info(f"📊 Adding {model_name} segment characteristics...")
        
        # Calculate segment characteristics from current data
        segment_stats = df_features.groupby(results[f'{model_name}_segment']).agg({
            'monetary_value': ['mean', 'count'],
            'frequency': 'mean',
            'engagement_score': 'mean'
        }).round(2)
        
        # Flatten column names
        segment_stats.columns = ['_'.join(col).strip() for col in segment_stats.columns]
        
        # Create segment mapping
        segment_map = {}
        for segment in segment_stats.index:
            segment_map[segment] = {
                'segment_type': f'Segment_{segment}',
                'avg_monetary_value': segment_stats.loc[segment, 'monetary_value_mean'],
                'avg_frequency': segment_stats.loc[segment, 'frequency_mean'],
                'customer_count': segment_stats.loc[segment, 'monetary_value_count']
            }
        
        # Add segment characteristics
        results[f'{model_name}_segment_type'] = results[f'{model_name}_segment'].map(
            lambda x: segment_map.get(x, {}).get('segment_type', 'Unknown')
        )
        results[f'{model_name}_avg_monetary'] = results[f'{model_name}_segment'].map(
            lambda x: segment_map.get(x, {}).get('avg_monetary_value', 0)
        )
        results[f'{model_name}_avg_frequency'] = results[f'{model_name}_segment'].map(
            lambda x: segment_map.get(x, {}).get('avg_frequency', 0)
        )
        results[f'{model_name}_segment_size'] = results[f'{model_name}_segment'].map(
            lambda x: segment_map.get(x, {}).get('customer_count', 0)
        )
        
        return results
    
    def generate_inference_report(self, results, model_name):
        """Generate inference report"""
        logger.info(f"📋 Generating {model_name} inference report...")
        
        # Segment distribution
        segment_dist = results[f'{model_name}_segment'].value_counts().sort_index()
        
        # Segment type distribution
        segment_type_dist = results[f'{model_name}_segment_type'].value_counts()
        
        # Average characteristics by segment
        segment_characteristics = results.groupby(f'{model_name}_segment').agg({
            'customer_id': 'count',
            f'{model_name}_avg_monetary': 'mean',
            f'{model_name}_avg_frequency': 'mean'
        }).rename(columns={'customer_id': 'customer_count'})
        
        report = {
            'inference_info': {
                'model_name': model_name,
                'inference_date': datetime.now().isoformat(),
                'total_customers': len(results),
                'segments_found': len(segment_dist)
            },
            'segment_distribution': segment_dist.to_dict(),
            'segment_type_distribution': segment_type_dist.to_dict(),
            'segment_characteristics': segment_characteristics.to_dict('index'),
            'model_metadata': {
                'training_date': self.metadata[model_name]['training_date'],
                'feature_count': len(self.metadata[model_name]['feature_columns']),
                'performance_metrics': {
                    'silhouette_score': 0.0,
                    'calinski_harabasz_score': 0.0
                }
            }
        }
        
        return report
    
    def save_inference_results(self, results, model_name, output_dir=None):
        """Save inference results"""
        logger.info(f"💾 Saving {model_name} inference results...")
        
        if output_dir is None:
            output_dir = os.path.join(self.config['ml']['model_storage_path'], 'customer_segmentation', 'inference_results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f'{model_name}_inference_results_{timestamp}.parquet')
        results.to_parquet(results_file, index=False)
        
        # Save report
        report = self.generate_inference_report(results, model_name)
        report_file = os.path.join(output_dir, f'{model_name}_inference_report_{timestamp}.yaml')
        with open(report_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f"✅ Results saved to {results_file}")
        logger.info(f"✅ Report saved to {report_file}")
        
        return results_file, report_file
    
    def create_inference_visualizations(self, results, model_name, output_dir=None):
        """Create inference visualizations"""
        logger.info(f"📊 Creating {model_name} inference visualizations...")
        
        if output_dir is None:
            output_dir = os.path.join(self.config['ml']['model_storage_path'], 'customer_segmentation', 'inference_results')
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Segment distribution
        fig1 = px.bar(
            x=results[f'{model_name}_segment'].value_counts().index,
            y=results[f'{model_name}_segment'].value_counts().values,
            title=f'{model_name.upper()} Segment Distribution',
            labels={'x': 'Segment ID', 'y': 'Customer Count'}
        )
        fig1.write_html(os.path.join(output_dir, f'{model_name}_segment_distribution_{timestamp}.html'))
        
        # Segment type distribution
        fig2 = px.pie(
            values=results[f'{model_name}_segment_type'].value_counts().values,
            names=results[f'{model_name}_segment_type'].value_counts().index,
            title=f'{model_name.upper()} Segment Type Distribution'
        )
        fig2.write_html(os.path.join(output_dir, f'{model_name}_segment_types_{timestamp}.html'))
        
        logger.info(f"✅ Visualizations saved to {output_dir}")
    
    def run_batch_inference(self, data_path=None, models=None):
        """Run batch inference for all models"""
        logger.info("🚀 Starting Customer Segmentation Batch Inference...")
        
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
                df_features = self.prepare_inference_features(df, model_name)
                
                # Perform inference
                results = self.perform_inference(df_features, model_name)
                
                # Save results
                results_file, report_file = self.save_inference_results(results, model_name)
                
                # Create visualizations
                self.create_inference_visualizations(results, model_name)
                
                all_results[model_name] = {
                    'results': results,
                    'results_file': results_file,
                    'report_file': report_file
                }
                
                logger.info(f"✅ {model_name} batch inference completed")
            
            logger.info("=" * 60)
            logger.info("🎉 BATCH INFERENCE COMPLETED!")
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
        inference = CustomerSegmentationBatchInference()
        results = inference.run_batch_inference()
        
        print(f"\n🎉 Customer Segmentation Batch Inference completed successfully!")
        print(f"📊 Results saved to models/customer_segmentation/inference_results/")
        print(f"📈 Ready for business analysis!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
