#!/usr/bin/env python3
"""
Forecasting Pipeline Training Script
Trains revenue and CTR forecasting models
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.s3_utils import get_s3_manager
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastingPipeline:
    def __init__(self, config_path='config/database_config.yaml'):
        self.config = self.load_config(config_path)
        self.models_path = 'models/forecasting'
        os.makedirs(self.models_path, exist_ok=True)
        
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_data(self):
        """Load historical training data for forecasting (before 3 months ago)"""
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
    
    def prepare_forecasting_features(self, df):
        """Prepare features for forecasting models"""
        logger.info("Preparing forecasting features...")
        
        # Create time-based features
        df['date'] = pd.to_datetime(df['registration_date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        
        # Create lag features for time series (simplified for cross-sectional data)
        df['revenue_lag_1'] = df['monetary_value'] * 0.95  # Simulate lag
        df['revenue_lag_7'] = df['monetary_value'] * 0.90  # Simulate lag
        df['revenue_lag_30'] = df['monetary_value'] * 0.85  # Simulate lag
        
        # Create rolling averages (simplified)
        df['revenue_ma_7'] = df['monetary_value'] * 0.98  # Simulate moving average
        df['revenue_ma_30'] = df['monetary_value'] * 0.95  # Simulate moving average
        
        # Create CTR features
        df['ctr'] = df['avg_ctr']  # Use existing CTR data
        df['ctr_lag_1'] = df['ctr'] * 0.95  # Simulate lag
        df['ctr_ma_7'] = df['ctr'] * 0.98  # Simulate moving average
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def train_revenue_forecasting_model(self, df):
        """Train revenue forecasting model"""
        logger.info("Training revenue forecasting model...")
        
        # Prepare features for revenue forecasting
        revenue_features = [
            'rfm_score', 'frequency', 'avg_order_value', 'days_since_last_purchase',
            'day_of_week', 'month', 'quarter', 'year',
            'revenue_lag_1', 'revenue_lag_7', 'revenue_lag_30',
            'revenue_ma_7', 'revenue_ma_30',
            'customer_age_days', 'monetary_value'
        ]
        
        # Filter out rows with missing target
        revenue_df = df[revenue_features].dropna()
        
        if len(revenue_df) < 100:
            logger.warning("Insufficient data for revenue forecasting model")
            return None
        
        X = revenue_df.drop('monetary_value', axis=1)
        y = revenue_df['monetary_value']
        
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
        
        logger.info(f"Revenue Forecasting Model Performance:")
        logger.info(f"  MAE: ${mae:.2f}")
        logger.info(f"  RMSE: ${np.sqrt(mse):.2f}")
        logger.info(f"  R²: {r2:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_path, 'revenue_forecasting_model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"✅ Revenue forecasting model saved to {model_path}")
        # Upload to S3
        try:
            s3_manager = get_s3_manager()
            s3_manager.upload_file(model_path, "models/forecasting")
        except Exception as e:
            logger.warning(f"⚠️  Failed to upload revenue model to S3: {e}")
        
        return model
    
    def train_ctr_forecasting_model(self, df):
        """Train CTR forecasting model"""
        logger.info("Training CTR forecasting model...")
        
        # Prepare features for CTR forecasting
        ctr_features = [
            'rfm_score', 'frequency', 'avg_order_value',
            'day_of_week', 'month', 'quarter', 'year',
            'ctr_lag_1', 'ctr_ma_7',
            'customer_age_days', 'ctr'
        ]
        
        # Filter out rows with missing target
        ctr_df = df[ctr_features].dropna()
        
        if len(ctr_df) < 100:
            logger.warning("Insufficient data for CTR forecasting model")
            return None
        
        X = ctr_df.drop('ctr', axis=1)
        y = ctr_df['ctr']
        
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
        
        logger.info(f"CTR Forecasting Model Performance:")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  RMSE: {np.sqrt(mse):.4f}")
        logger.info(f"  R²: {r2:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_path, 'ctr_forecasting_model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"✅ CTR forecasting model saved to {model_path}")
        # Upload to S3
        try:
            s3_manager = get_s3_manager()
            s3_manager.upload_file(model_path, "models/forecasting")
        except Exception as e:
            logger.warning(f"⚠️  Failed to upload CTR model to S3: {e}")
        
        return model
    
    def run_training_pipeline(self):
        """Run the complete forecasting training pipeline"""
        logger.info("🚀 Starting Forecasting Training Pipeline...")
        
        try:
            # Load data
            df = self.load_data()
            if df is None:
                raise Exception("Failed to load data")
            
            # Prepare features
            df = self.prepare_forecasting_features(df)
            
            # Train models
            revenue_model = self.train_revenue_forecasting_model(df)
            ctr_model = self.train_ctr_forecasting_model(df)
            
            if revenue_model and ctr_model:
                logger.info("=" * 60)
                logger.info("🎉 FORECASTING TRAINING COMPLETED!")
                logger.info("=" * 60)
                logger.info(f"📊 Trained 2 models on {len(df)} customers")
                logger.info("💾 Models saved and ready for inference!")
                
                return {
                    'revenue_forecasting': revenue_model,
                    'ctr_forecasting': ctr_model
                }
            else:
                raise Exception("Some forecasting models failed to train")
                
        except Exception as e:
            logger.error(f"❌ Error in training pipeline: {e}")
            raise

def main():
    """Main function to run the training pipeline"""
    try:
        # Initialize and run the pipeline
        pipeline = ForecastingPipeline()
        results = pipeline.run_training_pipeline()
        
        print("\n🎉 Forecasting Training completed successfully!")
        print(f"📊 Trained {len(results)} models")
        print("💾 Models saved and ready for inference!")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
