#!/usr/bin/env python3
"""
Journey Simulation Pipeline Training Script
Trains customer journey and conversion prediction models
"""

import pandas as pd
import numpy as np
import joblib
import logging
import yaml
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
import warnings

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.s3_utils import get_s3_manager
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JourneySimulationPipeline:
    def __init__(self, config_path='config/database_config.yaml'):
        self.config = self.load_config(config_path)
        self.models_path = 'models/journey_simulation'
        os.makedirs(self.models_path, exist_ok=True)
        
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_data(self):
        """Load historical training data for journey simulation (before 3 months ago)"""
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
    
    def prepare_journey_features(self, df):
        """Prepare features for journey simulation models"""
        logger.info("Preparing journey simulation features...")
        
        # Create journey stage features
        df['journey_stage'] = df.apply(self.calculate_journey_stage, axis=1)
        df['conversion_probability'] = df.apply(self.calculate_conversion_probability, axis=1)
        
        # Create behavioral features
        df['avg_session_duration'] = df['total_sessions'] / df['total_page_views'].replace(0, 1)
        df['bounce_rate'] = (df['total_sessions'] - df['total_page_views']) / df['total_sessions'].replace(0, 1)
        df['engagement_score'] = (df['total_page_views'] * 0.3 + df['total_events'] * 0.4 + df['total_interactions'] * 0.3)
        
        # Create time-based features
        df['days_since_first_visit'] = (pd.Timestamp.now() - pd.to_datetime(df['registration_date'])).dt.days
        df['visit_frequency'] = df['total_sessions'] / df['days_since_first_visit'].replace(0, 1)
        
        # Create product interaction features
        df['avg_product_views'] = df['product_views'] / df['total_sessions'].replace(0, 1)
        df['cart_add_rate'] = df['cart_adds'] / df['product_views'].replace(0, 1)
        df['purchase_rate'] = df['frequency'] / df['cart_adds'].replace(0, 1)
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def calculate_journey_stage(self, row):
        """Calculate customer journey stage"""
        if row['frequency'] > 0:
            if row['frequency'] >= 5:
                return 'loyal_customer'
            elif row['frequency'] >= 2:
                return 'repeat_customer'
            else:
                return 'first_time_buyer'
        else:
            if row['cart_adds'] > 0:
                return 'cart_abandoner'
            elif row['product_views'] > 0:
                return 'product_browser'
            else:
                return 'visitor'
    
    def calculate_conversion_probability(self, row):
        """Calculate conversion probability based on behavior"""
        base_prob = 0.1
        
        # RFM score influence
        if row['rfm_score'] > 80:
            base_prob += 0.3
        elif row['rfm_score'] > 60:
            base_prob += 0.2
        elif row['rfm_score'] > 40:
            base_prob += 0.1
        
        # Engagement influence
        if row['total_page_views'] > 10:
            base_prob += 0.1
        if row['total_events'] > 5:
            base_prob += 0.1
        if row['total_interactions'] > 3:
            base_prob += 0.1
        
        # Product interaction influence
        if row['product_views'] > 5:
            base_prob += 0.1
        if row['cart_adds'] > 0:
            base_prob += 0.2
        
        return min(base_prob, 0.95)
    
    def train_journey_stage_model(self, df):
        """Train journey stage prediction model"""
        logger.info("Training journey stage prediction model...")
        
        # Prepare features for journey stage prediction
        stage_features = [
            'rfm_score', 'frequency', 'avg_order_value', 'days_since_last_purchase',
            'total_sessions', 'total_page_views', 'total_events', 'total_interactions',
            'product_views', 'cart_adds', 'customer_age_days',
            'avg_session_duration', 'engagement_score',
            'purchase_velocity', 'session_intensity',
            'conversion_efficiency', 'search_intensity', 'product_exploration'
        ]
        
        # Filter out rows with missing target
        stage_df = df[stage_features + ['journey_stage']].dropna()
        
        if len(stage_df) < 100:
            logger.warning("Insufficient data for journey stage model")
            return None
        
        X = stage_df.drop('journey_stage', axis=1)
        y = stage_df['journey_stage']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        logger.info(f"Journey Stage Model Performance:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_path, 'journey_stage_model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"✅ Journey stage model saved to {model_path}")
        # Upload to S3
        try:
            s3_manager = get_s3_manager()
            s3_manager.upload_file(model_path, "models/journey_simulation")
        except Exception as e:
            logger.warning(f"⚠️  Failed to upload journey stage model to S3: {e}")
        
        return model
    
    def train_conversion_prediction_model(self, df):
        """Train conversion probability prediction model"""
        logger.info("Training conversion prediction model...")
        
        # Prepare features for conversion prediction
        conversion_features = [
            'rfm_score', 'frequency', 'avg_order_value', 'days_since_last_purchase',
            'total_sessions', 'total_page_views', 'total_events', 'total_interactions',
            'product_views', 'cart_adds', 'customer_age_days',
            'avg_session_duration', 'engagement_score',
            'purchase_velocity', 'session_intensity',
            'conversion_efficiency', 'search_intensity', 'product_exploration'
        ]
        
        # Filter out rows with missing target
        conversion_df = df[conversion_features + ['conversion_probability']].dropna()
        
        if len(conversion_df) < 100:
            logger.warning("Insufficient data for conversion prediction model")
            return None
        
        X = conversion_df.drop('conversion_probability', axis=1)
        y = conversion_df['conversion_probability']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Conversion Prediction Model Performance:")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  RMSE: {np.sqrt(mse):.4f}")
        logger.info(f"  R²: {r2:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_path, 'conversion_prediction_model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"✅ Conversion prediction model saved to {model_path}")
        # Upload to S3
        try:
            s3_manager = get_s3_manager()
            s3_manager.upload_file(model_path, "models/journey_simulation")
        except Exception as e:
            logger.warning(f"⚠️  Failed to upload conversion model to S3: {e}")
        
        return model
    
    def run_training_pipeline(self):
        """Run the complete journey simulation training pipeline"""
        logger.info("🚀 Starting Journey Simulation Training Pipeline...")
        
        try:
            # Load data
            df = self.load_data()
            if df is None:
                raise Exception("Failed to load data")
            
            # Prepare features
            df = self.prepare_journey_features(df)
            
            # Train models
            stage_model = self.train_journey_stage_model(df)
            conversion_model = self.train_conversion_prediction_model(df)
            
            if stage_model and conversion_model:
                logger.info("=" * 60)
                logger.info("🎉 JOURNEY SIMULATION TRAINING COMPLETED!")
                logger.info("=" * 60)
                logger.info(f"📊 Trained 2 models on {len(df)} customers")
                logger.info("💾 Models saved and ready for inference!")
                
                return {
                    'journey_stage': stage_model,
                    'conversion_prediction': conversion_model
                }
            else:
                raise Exception("Some journey simulation models failed to train")
                
        except Exception as e:
            logger.error(f"❌ Error in training pipeline: {e}")
            raise

def main():
    """Main function to run the training pipeline"""
    try:
        # Initialize and run the pipeline
        pipeline = JourneySimulationPipeline()
        results = pipeline.run_training_pipeline()
        
        print("\n🎉 Journey Simulation Training completed successfully!")
        print(f"📊 Trained {len(results)} models")
        print("💾 Models saved and ready for inference!")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
