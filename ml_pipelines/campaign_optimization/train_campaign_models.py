#!/usr/bin/env python3
"""
AMATO Production - Campaign Optimization ML Pipeline

This script trains models for campaign success prediction and budget optimization 
using customer behavioral data.

Author: Data Science Team
Date: 2024
"""

# Import required libraries
import pandas as pd
import numpy as np
import yaml
import logging
import os
import sys
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.s3_utils import get_s3_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CampaignOptimizationPipeline:
    def __init__(self, config_path='config/database_config.yaml'):
        self.config = self.load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        
    def load_config(self, config_path):
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return {}
    
    def load_data(self):
        """Load historical training data for campaign optimization (before 3 months ago)"""
        try:
            # Load historical training data from S3
            logger.info("🔍 Loading historical training data from S3...")
            s3_manager = get_s3_manager()
            s3_manager.load_training_data_from_s3()
            logger.info("✅ Historical training data loaded from S3")
            
            # Load the training dataset (historical data)
            data_path = 'data_pipelines/unified_dataset/output/unified_customer_dataset.parquet'
            
            if os.path.exists(data_path):
                df = pd.read_parquet(data_path)
                logger.info(f"✅ Loaded historical training dataset: {df.shape}")
                logger.info(f"📅 This dataset contains historical data for model training")
                return df
            else:
                logger.error(f"❌ Historical training dataset not found at {data_path}")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to load historical training data: {e}")
            return None
    
    def prepare_features(self, df, target_col):
        """Prepare features for campaign optimization"""
        logger.info(f"🔧 Preparing features for {target_col}...")
        
        # Select features based on target - use columns that actually exist
        if target_col == 'campaign_success':
            feature_columns = [
                'recency_days', 'frequency', 'monetary_value',
                'avg_order_value', 'customer_lifetime_value',
                'campaign_count', 'avg_roas', 'avg_ctr', 'total_campaign_revenue',
                'campaign_response_rate', 'avg_ctr_lift', 'rfm_score',
                'total_sessions', 'total_events',
                'conversion_probability', 'churn_risk', 'upsell_potential'
            ]
        elif target_col == 'budget_optimization':
            feature_columns = [
                'recency_days', 'frequency', 'monetary_value',
                'avg_order_value', 'customer_lifetime_value',
                'campaign_count', 'avg_roas', 'total_campaign_revenue',
                'rfm_score', 'conversion_probability'
            ]
        else:
            logger.error(f"❌ Unknown target column: {target_col}")
            return None, None, None
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        if len(available_features) < 5:
            logger.warning(f"⚠️  Only {len(available_features)} features available for {target_col}")
            
        # Create feature matrix
        X = df[available_features].copy()
        
        # Create synthetic target variables based on available data
        if target_col == 'campaign_success':
            # Create campaign success based on ROAS and CTR
            y = df.apply(lambda row: 
                'high' if row['avg_roas'] > 3.0 and row['avg_ctr'] > 0.05 else
                'medium' if row['avg_roas'] > 2.0 and row['avg_ctr'] > 0.03 else 'low', axis=1)
        elif target_col == 'budget_optimization':
            # Create budget optimization target based on campaign performance
            y = df.apply(lambda row: 
                min(1000, max(100, row['total_campaign_revenue'] * (row['avg_roas'] / 2.0))), axis=1)
        else:
            y = None
        
        # Handle missing values
        X = X.fillna(X.median())
        if y is not None:
            if y.dtype == 'object':  # Categorical target
                y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 'medium')
            else:  # Numeric target
                y = y.fillna(y.median())
        
        # Remove outliers using IQR method
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X[col] = X[col].clip(lower_bound, upper_bound)
        
        logger.info(f"✅ Prepared {len(X)} customers with {len(X.columns)} features for {target_col}")
        logger.info(f"   Target distribution: {y.value_counts().to_dict() if hasattr(y, 'value_counts') else 'Continuous'}")
        return X, y, available_features
    
    def train_campaign_success_model(self, X, y, n_estimators=100, max_depth=10):
        """Train campaign success prediction model"""
        logger.info(f"🎯 Training campaign success prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest classifier
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        logger.info(f"✅ Campaign success training completed")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   Precision: {precision:.4f}")
        logger.info(f"   Recall: {recall:.4f}")
        logger.info(f"   F1: {f1:.4f}")
        logger.info(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return model, scaler, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }
    
    def train_budget_optimization_model(self, X, y, n_estimators=100, max_depth=10):
        """Train budget optimization model"""
        logger.info(f"🎯 Training budget optimization model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest regressor
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        logger.info(f"✅ Budget optimization training completed")
        logger.info(f"   MSE: {mse:.6f}")
        logger.info(f"   MAE: {mae:.6f}")
        logger.info(f"   R²: {r2:.4f}")
        logger.info(f"   CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return model, scaler, {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }
    
    def save_models(self, success_model, success_scaler, budget_model, budget_scaler, 
                    success_metrics, budget_metrics, feature_names):
        """Save trained models and metadata"""
        logger.info("💾 Saving trained models...")
        
        # Create output directory
        output_dir = 'models/campaign_optimization'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save campaign success model
        success_path = os.path.join(output_dir, 'campaign_success_model.pkl')
        joblib.dump(success_model, success_path)
        
        # Save campaign success scaler
        success_scaler_path = os.path.join(output_dir, 'campaign_success_scaler.pkl')
        joblib.dump(success_scaler, success_scaler_path)
        
        # Save budget optimization model
        budget_path = os.path.join(output_dir, 'budget_optimization_model.pkl')
        joblib.dump(budget_model, budget_path)
        
        # Save budget optimization scaler
        budget_scaler_path = os.path.join(output_dir, 'budget_optimization_scaler.pkl')
        joblib.dump(budget_scaler, budget_scaler_path)
        
        # Save metadata
        metadata = {
            'campaign_success': success_metrics,
            'budget_optimization': budget_metrics,
            'feature_names': feature_names,
            'training_date': datetime.now().isoformat(),
            'model_versions': {
                'campaign_success': '1.0',
                'budget_optimization': '1.0'
            }
        }
        
        metadata_path = os.path.join(output_dir, 'pipeline_report.yaml')
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        # Upload to S3 directly (no local storage)
        try:
            s3_manager = get_s3_manager()
            
            # Upload models directly to S3
            success_uploaded = s3_manager.upload_model_direct(
                success_model, 'campaign_success_model', 'campaign_optimization', metadata
            )
            budget_uploaded = s3_manager.upload_model_direct(
                budget_model, 'budget_optimization_model', 'campaign_optimization', metadata
            )
            
            # Upload scalers directly to S3
            import io
            buffer = io.BytesIO()
            joblib.dump(success_scaler, buffer)
            success_scaler_bytes = buffer.getvalue()
            success_scaler_key = f"{s3_manager.base_path}/models/campaign_optimization/campaign_success_scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            success_scaler_uploaded = s3_manager.upload_bytes_direct(success_scaler_bytes, success_scaler_key)
            
            buffer = io.BytesIO()
            joblib.dump(budget_scaler, buffer)
            budget_scaler_bytes = buffer.getvalue()
            budget_scaler_key = f"{s3_manager.base_path}/models/campaign_optimization/budget_optimization_scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            budget_scaler_uploaded = s3_manager.upload_bytes_direct(budget_scaler_bytes, budget_scaler_key)
            
            # Upload metadata directly to S3
            metadata_bytes = yaml.dump(metadata, default_flow_style=False).encode('utf-8')
            metadata_key = f"{s3_manager.base_path}/models/campaign_optimization/pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            metadata_uploaded = s3_manager.upload_bytes_direct(metadata_bytes, metadata_key, 'text/yaml')
            
            # Check upload status
            if all([success_uploaded, budget_uploaded, success_scaler_uploaded, budget_scaler_uploaded, metadata_uploaded]):
                logger.info("✅ All models, scalers, and metadata uploaded directly to S3")
                return "S3://models/campaign_optimization/"
            else:
                logger.warning("⚠️  Some uploads failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to save models to S3: {e}")
            return None
    
    def run_training_pipeline(self):
        """Run the complete campaign optimization training pipeline"""
        logger.info("🚀 Starting Campaign Optimization Training Pipeline...")
        
        try:
            # Load data
            df = self.load_data()
            if df is None:
                raise Exception("Failed to load data")
            
            # Prepare features for campaign success
            X_success, y_success, success_features = self.prepare_features(df, 'campaign_success')
            if X_success is None:
                raise Exception("Failed to prepare campaign success features")
            
            # Prepare features for budget optimization
            X_budget, y_budget, budget_features = self.prepare_features(df, 'budget_optimization')
            if X_budget is None:
                raise Exception("Failed to prepare budget optimization features")
            
            # Train campaign success model
            success_model, success_scaler, success_metrics = self.train_campaign_success_model(X_success, y_success)
            
            # Train budget optimization model
            budget_model, budget_scaler, budget_metrics = self.train_budget_optimization_model(X_budget, y_budget)
            
            # Save models
            output_dir = self.save_models(
                success_model, success_scaler, budget_model, budget_scaler,
                success_metrics, budget_metrics, {
                    'campaign_success': success_features,
                    'budget_optimization': budget_features
                }
            )
            
            logger.info("=" * 60)
            logger.info("🎉 CAMPAIGN OPTIMIZATION TRAINING COMPLETED!")
            logger.info("=" * 60)
            logger.info(f"📊 Trained 2 models on {len(df)} customers")
            logger.info(f"🔧 Success features: {len(success_features)}, Budget features: {len(budget_features)}")
            logger.info(f"💾 Models saved to: {output_dir}")
            
            return {
                'campaign_success': success_model,
                'budget_optimization': budget_model,
                'success_metrics': success_metrics,
                'budget_metrics': budget_metrics
            }
            
        except Exception as e:
            logger.error(f"❌ Error in training pipeline: {e}")
            raise


def main():
    """Main function to run the training pipeline"""
    try:
        # Initialize and run the pipeline
        pipeline = CampaignOptimizationPipeline()
        results = pipeline.run_training_pipeline()
        
        print("\n🎉 Campaign Optimization Training completed successfully!")
        print(f"📊 Campaign Success: Accuracy = {results['success_metrics']['accuracy']:.4f}, F1 = {results['success_metrics']['f1']:.4f}")
        print(f"📊 Budget Optimization: R² = {results['budget_metrics']['r2']:.4f}, CV R² = {results['budget_metrics']['cv_r2_mean']:.4f}")
        print("💾 Models saved and ready for inference!")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
