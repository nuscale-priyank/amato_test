#!/usr/bin/env python3
"""
Campaign Optimization Pipeline Training Script
Trains campaign performance and A/B test optimization models
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

class CampaignOptimizationPipeline:
    def __init__(self, config_path='config/database_config.yaml'):
        self.config = self.load_config(config_path)
        self.models_path = 'models/campaign_optimization'
        os.makedirs(self.models_path, exist_ok=True)
        
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_unified_data(self):
        """Load unified dataset for campaign optimization"""
        try:
            # Load from parquet file
            data_path = 'data_pipelines/unified_dataset/output/unified_customer_dataset.parquet'
            if os.path.exists(data_path):
                df = pd.read_parquet(data_path)
                logger.info(f"Loaded unified dataset: {df.shape}")
                return df
            else:
                logger.error(f"Unified dataset not found: {data_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading unified data: {e}")
            return None
    
    def prepare_campaign_features(self, df):
        """Prepare features for campaign optimization models"""
        logger.info("Preparing campaign optimization features...")
        
        # Create campaign performance features
        df['campaign_roas'] = df['avg_roas']  # Use existing ROAS data
        df['campaign_cpa'] = df['total_campaign_revenue'] / df['campaign_count'].replace(0, 1)  # Simplified CPA
        df['campaign_ctr'] = df['avg_ctr']  # Use existing CTR data
        df['campaign_cvr'] = df['campaign_response_rate']  # Use response rate as conversion rate
        
        # Create campaign success indicators
        df['campaign_success'] = df.apply(self.calculate_campaign_success, axis=1)
        df['ab_test_winner'] = df.apply(self.calculate_ab_test_winner, axis=1)
        
        # Create customer-campaign interaction features
        df['customer_campaign_affinity'] = df.apply(self.calculate_customer_campaign_affinity, axis=1)
        df['optimal_campaign_budget'] = df.apply(self.calculate_optimal_budget, axis=1)
        
        # Create seasonal features
        df['campaign_month'] = pd.to_datetime(df['registration_date']).dt.month
        df['campaign_quarter'] = pd.to_datetime(df['registration_date']).dt.quarter
        df['campaign_day_of_week'] = pd.to_datetime(df['registration_date']).dt.dayofweek
        
        # Create targeting features
        df['target_audience_match'] = df.apply(self.calculate_target_audience_match, axis=1)
        df['channel_effectiveness'] = df.apply(self.calculate_channel_effectiveness, axis=1)
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def calculate_campaign_success(self, row):
        """Calculate campaign success based on multiple metrics"""
        success_score = 0
        
        # ROAS threshold
        if row['campaign_roas'] > 3.0:
            success_score += 3
        elif row['campaign_roas'] > 2.0:
            success_score += 2
        elif row['campaign_roas'] > 1.5:
            success_score += 1
        
        # CTR threshold
        if row['campaign_ctr'] > 0.05:
            success_score += 2
        elif row['campaign_ctr'] > 0.03:
            success_score += 1
        
        # Conversion rate threshold
        if row['campaign_cvr'] > 0.03:
            success_score += 2
        elif row['campaign_cvr'] > 0.02:
            success_score += 1
        
        # Budget efficiency
        if row['campaign_count'] < 5 and row['total_campaign_revenue'] > 2000:
            success_score += 1
        
        return 'high' if success_score >= 6 else 'medium' if success_score >= 3 else 'low'
    
    def calculate_ab_test_winner(self, row):
        """Calculate A/B test winner based on performance"""
        # Simplified A/B test winner calculation
        return 'A' if row['campaign_ctr'] > 0.04 and row['campaign_cvr'] > 0.02 else 'B'
    
    def calculate_customer_campaign_affinity(self, row):
        """Calculate customer-campaign affinity score"""
        affinity = 0.5  # Base affinity
        
        # RFM score influence
        if row['rfm_score'] > 80:
            affinity += 0.2
        elif row['rfm_score'] > 60:
            affinity += 0.1
        
        # Campaign type match
        if row['campaign_type'] in ['Email', 'Social Media'] and row['total_sessions'] > 10:
            affinity += 0.1
        
        # Channel preference (simplified)
        if row['total_events'] > 5:
            affinity += 0.1
        
        return min(affinity, 1.0)
    
    def calculate_optimal_budget(self, row):
        """Calculate optimal campaign budget based on customer value"""
        base_budget = 100
        
        # Customer value influence
        if row['total_campaign_revenue'] > 1000:
            base_budget += 200
        elif row['total_campaign_revenue'] > 500:
            base_budget += 100
        
        # RFM score influence
        if row['rfm_score'] > 80:
            base_budget *= 1.5
        elif row['rfm_score'] > 60:
            base_budget *= 1.2
        
        # Campaign type influence
        if row['campaign_type'] == 'Email':
            base_budget *= 0.8
        elif row['campaign_type'] == 'Video':
            base_budget *= 1.3
        
        return min(base_budget, 1000)
    
    def calculate_target_audience_match(self, row):
        """Calculate target audience match score"""
        match_score = 0.5
        
        # Age group match
        if row['target_audience'] == 'New Customers' and row['customer_age_days'] < 30:
            match_score += 0.2
        elif row['target_audience'] == 'Loyal' and row['total_transactions'] > 5:
            match_score += 0.2
        
        # High value customer match
        if row['target_audience'] == 'High Value' and row['total_campaign_revenue'] > 500:
            match_score += 0.2
        
        return min(match_score, 1.0)
    
    def calculate_channel_effectiveness(self, row):
        """Calculate channel effectiveness score"""
        effectiveness = 0.5
        
        # Simplified channel effectiveness based on performance metrics
        if row['campaign_ctr'] > 0.03:
            effectiveness += 0.2
        if row['campaign_cvr'] > 0.02:
            effectiveness += 0.2
        if row['campaign_roas'] > 2.0:
            effectiveness += 0.1
        
        return min(effectiveness, 1.0)
    
    def train_campaign_success_model(self, df):
        """Train campaign success prediction model"""
        logger.info("Training campaign success prediction model...")
        
        # Prepare features for campaign success prediction
        success_features = [
            'rfm_score', 'frequency', 'avg_order_value', 'days_since_last_purchase',
            'campaign_count', 'total_campaign_revenue', 'avg_roas', 'avg_ctr',
            'campaign_roas', 'campaign_cpa', 'campaign_ctr', 'campaign_cvr',
            'customer_campaign_affinity', 'target_audience_match', 'channel_effectiveness',
            'campaign_month', 'campaign_quarter', 'campaign_day_of_week'
        ]
        
        # Filter out rows with missing target
        success_df = df[success_features + ['campaign_success']].dropna()
        
        if len(success_df) < 100:
            logger.warning("Insufficient data for campaign success model")
            return None
        
        X = success_df.drop('campaign_success', axis=1)
        y = success_df['campaign_success']
        
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
        
        logger.info(f"Campaign Success Model Performance:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_path, 'campaign_success_model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"✅ Campaign success model saved to {model_path}")
        # Upload to S3
        try:
            s3_manager = get_s3_manager()
            s3_manager.upload_file(model_path, "models/campaign_optimization")
        except Exception as e:
            logger.warning(f"⚠️  Failed to upload campaign success model to S3: {e}")
        
        return model
    
    def train_budget_optimization_model(self, df):
        """Train budget optimization model"""
        logger.info("Training budget optimization model...")
        
        # Prepare features for budget optimization
        budget_features = [
            'rfm_score', 'frequency', 'avg_order_value', 'days_since_last_purchase',
            'campaign_count', 'total_campaign_revenue', 'avg_roas', 'avg_ctr',
            'campaign_roas', 'campaign_cpa', 'campaign_ctr', 'campaign_cvr',
            'customer_campaign_affinity', 'target_audience_match', 'channel_effectiveness',
            'campaign_month', 'campaign_quarter', 'campaign_day_of_week'
        ]
        
        # Filter out rows with missing target
        budget_df = df[budget_features + ['optimal_campaign_budget']].dropna()
        
        if len(budget_df) < 100:
            logger.warning("Insufficient data for budget optimization model")
            return None
        
        X = budget_df.drop('optimal_campaign_budget', axis=1)
        y = budget_df['optimal_campaign_budget']
        
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
        
        logger.info(f"Budget Optimization Model Performance:")
        logger.info(f"  MAE: ${mae:.2f}")
        logger.info(f"  RMSE: ${np.sqrt(mse):.2f}")
        logger.info(f"  R²: {r2:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_path, 'budget_optimization_model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"✅ Budget optimization model saved to {model_path}")
        # Upload to S3
        try:
            s3_manager = get_s3_manager()
            s3_manager.upload_file(model_path, "models/campaign_optimization")
        except Exception as e:
            logger.warning(f"⚠️  Failed to upload budget optimization model to S3: {e}")
        
        return model
    
    def train_all_models(self):
        """Train all campaign optimization models"""
        logger.info("🚀 Starting Campaign Optimization Pipeline Training...")
        
        # Load data
        df = self.load_unified_data()
        if df is None:
            logger.error("Failed to load unified data")
            return False
        
        # Prepare features
        df = self.prepare_campaign_features(df)
        
        # Train models
        success_model = self.train_campaign_success_model(df)
        budget_model = self.train_budget_optimization_model(df)
        
        if success_model and budget_model:
            logger.info("✅ All campaign optimization models trained successfully!")
            return True
        else:
            logger.error("❌ Some campaign optimization models failed to train")
            return False

if __name__ == "__main__":
    pipeline = CampaignOptimizationPipeline()
    pipeline.train_all_models()
