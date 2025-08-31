#!/usr/bin/env python3
"""
AMATO Production - Customer Segmentation ML Pipeline
Trains clustering models for customer segmentation using RFM and behavioral data
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
import joblib
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
import hdbscan
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerSegmentationPipeline:
    def __init__(self, config_path='config/database_config.yaml'):
        self.config = self.load_config(config_path)
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_unified_dataset(self):
        """Load the unified customer dataset"""
        logger.info("📊 Loading unified customer dataset...")
        
        dataset_path = os.path.join(
            self.config['data_pipelines']['parquet_storage']['base_path'],
            self.config['data_pipelines']['parquet_storage']['unified_customer']
        )
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Unified dataset not found at {dataset_path}")
        
        df = pd.read_parquet(dataset_path)
        logger.info(f"✅ Loaded dataset with {len(df)} customers and {len(df.columns)} features")
        return df
    
    def prepare_features(self, df):
        """Prepare features for segmentation"""
        logger.info("🔧 Preparing features for segmentation...")
        
        # Select relevant features for segmentation
        feature_columns = [
            # RFM Features
            'recency_days', 'frequency', 'monetary_value',
            'recency_score', 'frequency_score', 'monetary_score',
            'rfm_score', 'customer_lifetime_value',
            
            # Behavioral Features
            'total_sessions', 'avg_session_duration', 'total_page_views',
            'conversion_events', 'add_to_cart_events', 'search_queries',
            'product_interactions', 'unique_products_viewed',
            'avg_engagement_score', 'cart_abandonment_rate',
            'bounce_rate', 'return_visitor_rate',
            
            # Engineered Features
            'purchase_velocity', 'engagement_score_normalized',
            'session_intensity', 'conversion_efficiency',
            'search_intensity', 'product_exploration',
            'cart_behavior', 'rfm_composite',
            'value_engagement_ratio', 'churn_risk',
            'upsell_potential', 'lifetime_value_potential',
            
            # Binary Features
            'is_high_value', 'is_engaged', 'is_mobile_user', 'is_return_visitor'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = [col for col in feature_columns if col not in df.columns]
        
        if missing_features:
            logger.warning(f"⚠️ Missing features: {missing_features}")
        
        # Select features and handle missing values
        df_features = df[['customer_id'] + available_features].copy()
        df_features = df_features.fillna(0)
        
        # Remove customers with no purchase history for better segmentation
        df_features = df_features[df_features['monetary_value'] > 0]
        
        self.feature_columns = available_features
        logger.info(f"✅ Prepared {len(available_features)} features for {len(df_features)} customers")
        
        return df_features
    
    def preprocess_data(self, df_features):
        """Preprocess data for clustering"""
        logger.info("🔄 Preprocessing data...")
        
        # Separate features and customer IDs
        customer_ids = df_features['customer_id'].values
        X = df_features[self.feature_columns].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"✅ Preprocessed data: {X_scaled.shape}")
        return X_scaled, customer_ids
    
    def train_kmeans_model(self, X_scaled, n_clusters=5):
        """Train K-means clustering model"""
        logger.info(f"🎯 Training K-means model with {n_clusters} clusters...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        kmeans.fit(X_scaled)
        
        # Evaluate model
        silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
        calinski_avg = calinski_harabasz_score(X_scaled, kmeans.labels_)
        
        logger.info(f"✅ K-means trained - Silhouette: {silhouette_avg:.3f}, Calinski-Harabasz: {calinski_avg:.3f}")
        
        return kmeans, silhouette_avg, calinski_avg
    
    def train_hdbscan_model(self, X_scaled):
        """Train HDBSCAN clustering model"""
        logger.info("🎯 Training HDBSCAN model...")
        
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=10,
            cluster_selection_epsilon=0.1,
            cluster_selection_method='eom'
        )
        
        hdbscan_model.fit(X_scaled)
        
        # Evaluate model (only if clusters found)
        if len(set(hdbscan_model.labels_)) > 1:
            silhouette_avg = silhouette_score(X_scaled, hdbscan_model.labels_)
            calinski_avg = calinski_harabasz_score(X_scaled, hdbscan_model.labels_)
        else:
            silhouette_avg = 0
            calinski_avg = 0
        
        n_clusters = len(set(hdbscan_model.labels_)) - (1 if -1 in hdbscan_model.labels_ else 0)
        n_noise = list(hdbscan_model.labels_).count(-1)
        
        logger.info(f"✅ HDBSCAN trained - Clusters: {n_clusters}, Noise points: {n_noise}")
        logger.info(f"   Silhouette: {silhouette_avg:.3f}, Calinski-Harabasz: {calinski_avg:.3f}")
        
        return hdbscan_model, silhouette_avg, calinski_avg
    
    def analyze_segments(self, df_features, model, model_name):
        """Analyze and characterize segments"""
        logger.info(f"📊 Analyzing segments for {model_name}...")
        
        # Get predictions
        X = df_features[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Add predictions to dataframe
        df_analysis = df_features.copy()
        df_analysis[f'{model_name}_segment'] = predictions
        
        # Segment analysis
        segment_analysis = df_analysis.groupby(f'{model_name}_segment').agg({
            'monetary_value': ['mean', 'std', 'count'],
            'frequency': ['mean', 'std'],
            'recency_days': ['mean', 'std'],
            'avg_engagement_score': ['mean', 'std'],
            'customer_lifetime_value': ['mean', 'std'],
            'churn_risk': ['mean', 'std'],
            'upsell_potential': ['mean', 'std']
        }).round(2)
        
        # Flatten column names
        segment_analysis.columns = ['_'.join(col).strip() for col in segment_analysis.columns]
        
        # Add segment characteristics
        segment_characteristics = []
        for segment in df_analysis[f'{model_name}_segment'].unique():
            segment_data = df_analysis[df_analysis[f'{model_name}_segment'] == segment]
            
            # Determine segment type based on characteristics
            avg_monetary = segment_data['monetary_value'].mean()
            avg_frequency = segment_data['frequency'].mean()
            avg_recency = segment_data['recency_days'].mean()
            avg_engagement = segment_data['avg_engagement_score'].mean()
            
            if avg_monetary > 2000 and avg_frequency > 10:
                segment_type = "High-Value Loyal"
            elif avg_monetary > 1000 and avg_recency < 30:
                segment_type = "High-Value Recent"
            elif avg_frequency > 5 and avg_engagement > 70:
                segment_type = "Engaged Regular"
            elif avg_recency < 60 and avg_engagement > 50:
                segment_type = "Recent Engaged"
            elif avg_recency > 180:
                segment_type = "At Risk"
            else:
                segment_type = "Standard"
            
            segment_characteristics.append({
                'segment': segment,
                'segment_type': segment_type,
                'customer_count': len(segment_data),
                'avg_monetary_value': avg_monetary,
                'avg_frequency': avg_frequency,
                'avg_recency_days': avg_recency,
                'avg_engagement_score': avg_engagement
            })
        
        segment_summary = pd.DataFrame(segment_characteristics)
        
        logger.info(f"✅ Segment analysis completed for {model_name}")
        return df_analysis, segment_analysis, segment_summary
    
    def create_visualizations(self, df_analysis, model_name):
        """Create segment visualizations"""
        logger.info(f"📈 Creating visualizations for {model_name}...")
        
        # Create output directory
        output_dir = os.path.join(self.config['ml']['model_storage_path'], 'customer_segmentation', 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Segment distribution
        fig_dist = px.histogram(
            df_analysis, 
            x=f'{model_name}_segment',
            title=f'{model_name} - Segment Distribution',
            labels={f'{model_name}_segment': 'Segment', 'count': 'Customer Count'}
        )
        fig_dist.write_html(os.path.join(output_dir, f'{model_name}_segment_distribution.html'))
        
        # 2. RFM scatter plot
        fig_rfm = px.scatter(
            df_analysis,
            x='recency_days',
            y='monetary_value',
            color=f'{model_name}_segment',
            size='frequency',
            hover_data=['customer_id'],
            title=f'{model_name} - RFM Scatter Plot',
            labels={'recency_days': 'Recency (Days)', 'monetary_value': 'Monetary Value ($)'}
        )
        fig_rfm.write_html(os.path.join(output_dir, f'{model_name}_rfm_scatter.html'))
        
        # 3. Segment characteristics radar chart
        segment_means = df_analysis.groupby(f'{model_name}_segment').agg({
            'monetary_value': 'mean',
            'frequency': 'mean',
            'avg_engagement_score': 'mean',
            'customer_lifetime_value': 'mean',
            'purchase_velocity': 'mean'
        }).reset_index()
        
        # Normalize values for radar chart
        for col in ['monetary_value', 'frequency', 'avg_engagement_score', 'customer_lifetime_value', 'purchase_velocity']:
            segment_means[f'{col}_normalized'] = (segment_means[col] - segment_means[col].min()) / (segment_means[col].max() - segment_means[col].min())
        
        fig_radar = go.Figure()
        
        for _, row in segment_means.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['monetary_value_normalized'], row['frequency_normalized'], 
                   row['avg_engagement_score_normalized'], row['customer_lifetime_value_normalized'],
                   row['purchase_velocity_normalized']],
                theta=['Monetary', 'Frequency', 'Engagement', 'CLV', 'Velocity'],
                fill='toself',
                name=f'Segment {row[f"{model_name}_segment"]}'
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=f'{model_name} - Segment Characteristics Radar Chart'
        )
        fig_radar.write_html(os.path.join(output_dir, f'{model_name}_segment_radar.html'))
        
        logger.info(f"✅ Visualizations saved to {output_dir}")
    
    def save_models(self, models_data):
        """Save trained models and metadata"""
        logger.info("💾 Saving models and metadata...")
        
        # Create output directory
        output_dir = os.path.join(self.config['ml']['model_storage_path'], 'customer_segmentation')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model_data in models_data.items():
            model_file = os.path.join(output_dir, f'{model_name}_model.pkl')
            joblib.dump(model_data['model'], model_file)
            
            # Save scaler
            scaler_file = os.path.join(output_dir, f'{model_name}_scaler.pkl')
            joblib.dump(self.scaler, scaler_file)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'model_type': type(model_data['model']).__name__,
                'feature_columns': self.feature_columns,
                'training_date': datetime.now().isoformat(),
                'performance_metrics': {
                    'silhouette_score': model_data['silhouette_score'],
                    'calinski_harabasz_score': model_data['calinski_score']
                },
                'segment_summary': model_data['segment_summary'].to_dict('records')
            }
            
            metadata_file = os.path.join(output_dir, f'{model_name}_metadata.yaml')
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
            
            logger.info(f"✅ Saved {model_name} model and metadata")
        
        # Save feature importance/characteristics
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'mean': self.scaler.mean_,
            'std': self.scaler.scale_
        })
        feature_importance.to_csv(os.path.join(output_dir, 'feature_characteristics.csv'), index=False)
        
        logger.info(f"✅ All models saved to {output_dir}")
    
    def run_pipeline(self):
        """Run the complete customer segmentation pipeline"""
        logger.info("🚀 Starting Customer Segmentation Pipeline...")
        
        try:
            # Load data
            df = self.load_unified_dataset()
            
            # Prepare features
            df_features = self.prepare_features(df)
            
            # Preprocess data
            X_scaled, customer_ids = self.preprocess_data(df_features)
            
            # Train models
            models_data = {}
            
            # K-means model
            kmeans_model, kmeans_silhouette, kmeans_calinski = self.train_kmeans_model(X_scaled, n_clusters=5)
            df_kmeans, kmeans_analysis, kmeans_summary = self.analyze_segments(df_features, kmeans_model, 'kmeans')
            self.create_visualizations(df_kmeans, 'kmeans')
            
            models_data['kmeans'] = {
                'model': kmeans_model,
                'silhouette_score': kmeans_silhouette,
                'calinski_score': kmeans_calinski,
                'segment_analysis': kmeans_analysis,
                'segment_summary': kmeans_summary
            }
            
            # HDBSCAN model
            hdbscan_model, hdbscan_silhouette, hdbscan_calinski = self.train_hdbscan_model(X_scaled)
            df_hdbscan, hdbscan_analysis, hdbscan_summary = self.analyze_segments(df_features, hdbscan_model, 'hdbscan')
            self.create_visualizations(df_hdbscan, 'hdbscan')
            
            models_data['hdbscan'] = {
                'model': hdbscan_model,
                'silhouette_score': hdbscan_silhouette,
                'calinski_score': hdbscan_calinski,
                'segment_analysis': hdbscan_analysis,
                'segment_summary': hdbscan_summary
            }
            
            # Save models
            self.save_models(models_data)
            
            # Generate final report
            self.generate_final_report(models_data)
            
            logger.info("=" * 60)
            logger.info("🎉 CUSTOMER SEGMENTATION PIPELINE COMPLETED!")
            logger.info("=" * 60)
            logger.info(f"📊 Trained {len(models_data)} models")
            logger.info(f"🎯 K-means: {len(kmeans_summary)} segments, Silhouette: {kmeans_silhouette:.3f}")
            logger.info(f"🎯 HDBSCAN: {len(hdbscan_summary)} segments, Silhouette: {hdbscan_silhouette:.3f}")
            logger.info(f"💾 Models saved to: {self.config['ml']['model_storage_path']}/customer_segmentation")
            
            return models_data
            
        except Exception as e:
            logger.error(f"❌ Error in segmentation pipeline: {e}")
            raise
    
    def generate_final_report(self, models_data):
        """Generate final pipeline report"""
        logger.info("📋 Generating final report...")
        
        report = {
            'pipeline_info': {
                'pipeline_name': 'Customer Segmentation',
                'execution_date': datetime.now().isoformat(),
                'total_customers': len(self.feature_columns),
                'total_features': len(self.feature_columns)
            },
            'models_trained': {
                model_name: {
                    'model_type': type(model_data['model']).__name__,
                    'silhouette_score': model_data['silhouette_score'],
                    'calinski_harabasz_score': model_data['calinski_score'],
                    'segments_found': len(model_data['segment_summary'])
                }
                for model_name, model_data in models_data.items()
            },
            'feature_importance': {
                'feature_columns': self.feature_columns,
                'scaling_applied': True
            },
            'recommendations': {
                'best_model': max(models_data.keys(), key=lambda x: models_data[x]['silhouette_score']),
                'optimal_segments': '5-7 segments recommended for business use',
                'next_steps': [
                    'Validate segments with business stakeholders',
                    'Create segment-specific marketing strategies',
                    'Monitor segment evolution over time',
                    'Implement automated segment assignment'
                ]
            }
        }
        
        # Save report
        report_file = os.path.join(
            self.config['ml']['model_storage_path'], 
            'customer_segmentation', 
            'pipeline_report.yaml'
        )
        with open(report_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f"✅ Final report saved to {report_file}")

def main():
    """Main function"""
    try:
        pipeline = CustomerSegmentationPipeline()
        models_data = pipeline.run_pipeline()
        
        print(f"\n🎉 Customer Segmentation Pipeline completed successfully!")
        print(f"📊 Trained {len(models_data)} models")
        print(f"💾 Models saved to: models/customer_segmentation/")
        print(f"📈 Ready for inference and deployment!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
