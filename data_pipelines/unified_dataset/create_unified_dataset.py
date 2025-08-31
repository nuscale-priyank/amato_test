#!/usr/bin/env python3
"""
AMATO Production - Unified Dataset Creation
This script combines all transformed data from Trino into a unified customer dataset for ML consumption
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedDatasetCreator:
    def __init__(self, config_path='config/database_config.yaml'):
        self.config = self.load_config(config_path)
        self.trino_engine = None
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def connect_trino(self):
        """Connect to Trino"""
        # Trino connection
        trino_config = self.config['trino']
        trino_connection_string = f"trino://{trino_config['username']}:{trino_config['password']}@{trino_config['host']}:{trino_config['port']}"
        self.trino_engine = create_engine(trino_connection_string)
        
        logger.info("✅ Connected to Trino")
    
    def load_customer_rfm_data(self):
        """Load customer RFM data from Trino"""
        logger.info("📊 Loading customer RFM data from Trino...")
        
        query = """
        SELECT 
            customer_id,
            email,
            first_name,
            last_name,
            registration_date,
            recency_days,
            frequency,
            monetary_value,
            recency_score,
            frequency_score,
            monetary_score,
            rfm_score,
            rfm_segment,
            customer_value_tier,
            engagement_level,
            total_items_purchased,
            avg_order_value,
            unique_purchase_days,
            customer_lifetime_value,
            purchase_frequency_rate
        FROM nucentral_mysqldb.amato.customer_rfm_data
        """
        
        df_rfm = pd.read_sql(query, self.trino_engine)
        logger.info(f"✅ Loaded {len(df_rfm)} customer RFM records")
        return df_rfm
    
    def load_customer_demographics(self):
        """Load customer demographics data from Trino"""
        logger.info("👥 Loading customer demographics data from Trino...")
        
        query = """
        SELECT 
            cd.customer_id,
            cd.age_group,
            cd.education_level,
            cd.occupation,
            cd.household_size,
            cd.marital_status,
            cd.interests,
            cd.lifestyle_segment
        FROM nucentral_mysqldb.amato.customer_demographics_clean cd
        INNER JOIN nucentral_mysqldb.amato.customers_clean c ON cd.customer_id = c.customer_id
        """
        
        df_demographics = pd.read_sql(query, self.trino_engine)
        logger.info(f"✅ Loaded {len(df_demographics)} customer demographics records")
        return df_demographics
    
    def load_campaign_performance_data(self):
        """Load campaign performance data from Trino"""
        logger.info("📢 Loading campaign performance data from Trino...")
        
        query = """
        SELECT 
            campaign_id,
            campaign_name,
            campaign_type,
            channel,
            target_audience,
            start_date,
            end_date,
            budget,
            status,
            total_impressions,
            total_clicks,
            total_conversions,
            total_revenue,
            overall_ctr,
            overall_conversion_rate,
            overall_roas,
            performance_segment,
            efficiency_score,
            budget_utilization_percent,
            revenue_rank,
            conversion_rank,
            roas_rank
        FROM nucentral_postgresdb.amato.campaign_performance_data
        """
        
        df_campaigns = pd.read_sql(query, self.trino_engine)
        logger.info(f"✅ Loaded {len(df_campaigns)} campaign performance records")
        return df_campaigns
    
    def load_ab_test_results(self):
        """Load A/B test results from Trino"""
        logger.info("🧪 Loading A/B test results from Trino...")
        
        query = """
        SELECT 
            test_id,
            test_name,
            campaign_id,
            variant_a_description,
            variant_b_description,
            start_date,
            end_date,
            test_duration_days,
            test_status,
            variant_a_ctr,
            variant_b_ctr,
            variant_a_conversion_rate,
            variant_b_conversion_rate,
            ctr_lift_percentage,
            conversion_lift_percentage,
            revenue_lift_percentage,
            winner,
            confidence_level,
            statistical_significance,
            recommendation,
            p_value,
            sample_size_required
        FROM nucentral_postgresdb.amato.ab_test_results_data
        """
        
        df_ab_tests = pd.read_sql(query, self.trino_engine)
        logger.info(f"✅ Loaded {len(df_ab_tests)} A/B test records")
        return df_ab_tests
    
    def load_customer_journey_data(self):
        """Load customer journey data from Trino"""
        logger.info("🛤️ Loading customer journey data from Trino...")
        
        query = """
        SELECT 
            customer_id,
            email,
            first_name,
            last_name,
            total_sessions,
            avg_session_duration,
            device_preference,
            browser_preference,
            total_page_views,
            unique_pages_viewed,
            avg_time_on_page,
            scroll_engagement,
            total_events,
            click_events,
            add_to_cart_events,
            purchase_events,
            event_category,
            total_interactions,
            unique_products_viewed,
            product_views,
            cart_adds,
            interaction_intensity,
            total_search_queries,
            unique_search_terms,
            clicked_search_results,
            search_effectiveness,
            journey_type,
            conversion_status,
            journey_complexity,
            engagement_score,
            conversion_probability
        FROM nucentral_mysqldb.amato.customer_journey_data
        """
        
        df_journey = pd.read_sql(query, self.trino_engine)
        logger.info(f"✅ Loaded {len(df_journey)} customer journey records")
        return df_journey
    
    def create_customer_campaign_mapping(self, df_campaigns, df_ab_tests):
        """Create customer-campaign mapping based on target audience"""
        logger.info("🔗 Creating customer-campaign mapping...")
        
        # Create campaign summary by target audience
        campaign_summary = df_campaigns.groupby('target_audience').agg({
            'campaign_id': 'count',
            'overall_roas': 'mean',
            'overall_ctr': 'mean',
            'overall_conversion_rate': 'mean',
            'total_revenue': 'sum'
        }).reset_index()
        
        campaign_summary.columns = [
            'target_audience', 'campaign_count', 'avg_roas', 
            'avg_ctr', 'avg_conversion_rate', 'total_campaign_revenue'
        ]
        
        # Create A/B test summary
        ab_test_summary = df_ab_tests.groupby('campaign_id').agg({
            'test_id': 'count',
            'ctr_lift_percentage': 'mean',
            'conversion_lift_percentage': 'mean',
            'statistical_significance': lambda x: (x == 'Yes').sum()
        }).reset_index()
        
        ab_test_summary.columns = [
            'campaign_id', 'ab_test_count', 'avg_ctr_lift', 
            'avg_conversion_lift', 'significant_tests'
        ]
        
        return campaign_summary, ab_test_summary
    
    def create_unified_dataset(self):
        """Create the unified customer dataset"""
        logger.info("🚀 Creating unified customer dataset...")
        
        # Load all data from Trino
        df_rfm = self.load_customer_rfm_data()
        df_demographics = self.load_customer_demographics()
        df_campaigns = self.load_campaign_performance_data()
        df_ab_tests = self.load_ab_test_results()
        df_journey = self.load_customer_journey_data()
        
        # Create campaign mappings
        campaign_summary, ab_test_summary = self.create_customer_campaign_mapping(df_campaigns, df_ab_tests)
        
        # Start with RFM data as the base
        df_unified = df_rfm.copy()
        
        # Merge demographics
        df_unified = df_unified.merge(df_demographics, on='customer_id', how='left')
        
        # Merge journey data
        df_unified = df_unified.merge(df_journey, on='customer_id', how='left', suffixes=('', '_journey'))
        
        # Add campaign performance metrics based on customer segments
        # Map customer segments to target audiences
        segment_to_audience = {
            'Champions': 'High Value',
            'Loyal': 'Existing Customers',
            'At Risk': 'At Risk',
            'Can\'t Lose': 'High Value',
            'New': 'New Customers',
            'Promising': 'New Customers'
        }
        
        df_unified['target_audience'] = df_unified['rfm_segment'].map(segment_to_audience)
        df_unified = df_unified.merge(campaign_summary, on='target_audience', how='left')
        
        # Add A/B test performance (aggregated by campaign type)
        campaign_ab_summary = df_ab_tests.merge(
            df_campaigns[['campaign_id', 'campaign_type']], on='campaign_id', how='left'
        ).groupby('campaign_type').agg({
            'ctr_lift_percentage': 'mean',
            'conversion_lift_percentage': 'mean',
            'statistical_significance': lambda x: (x == 'Yes').sum()
        }).reset_index()
        
        campaign_ab_summary.columns = [
            'campaign_type', 'avg_ctr_lift', 'avg_conversion_lift', 'significant_ab_tests'
        ]
        
        # Map campaign types to customer segments
        segment_to_campaign_type = {
            'Champions': 'Email',
            'Loyal': 'Social Media',
            'At Risk': 'Display',
            'Can\'t Lose': 'Email',
            'New': 'Search',
            'Promising': 'Social Media'
        }
        
        df_unified['campaign_type'] = df_unified['rfm_segment'].map(segment_to_campaign_type)
        df_unified = df_unified.merge(campaign_ab_summary, on='campaign_type', how='left')
        
        # Feature engineering
        df_unified = self.engineer_features(df_unified)
        
        # Clean and finalize
        df_unified = self.clean_unified_dataset(df_unified)
        
        logger.info(f"✅ Created unified dataset with {len(df_unified)} customers and {len(df_unified.columns)} features")
        return df_unified
    
    def engineer_features(self, df):
        """Engineer additional features for ML"""
        logger.info("🔧 Engineering features...")
        
        # Customer lifecycle features
        df['customer_age_days'] = (pd.Timestamp.now() - pd.to_datetime(df['registration_date'])).dt.days
        df['days_since_last_purchase'] = df['recency_days']
        df['purchase_velocity'] = df['frequency'] / (df['customer_age_days'] + 1)
        
        # Engagement features
        df['engagement_score_normalized'] = df['engagement_score'] / 100
        df['session_intensity'] = df['total_page_views'] / (df['total_sessions'] + 1)
        df['conversion_efficiency'] = df['purchase_events'] / (df['total_sessions'] + 1)
        
        # Behavioral features
        df['search_intensity'] = df['total_search_queries'] / (df['total_sessions'] + 1)
        df['product_exploration'] = df['unique_products_viewed'] / (df['total_sessions'] + 1)
        df['cart_behavior'] = df['cart_adds'] / (df['total_sessions'] + 1)
        
        # RFM composite features
        df['rfm_composite'] = (df['recency_score'] + df['frequency_score'] + df['monetary_score']) / 3
        df['value_engagement_ratio'] = df['monetary_value'] / (df['engagement_score'] + 1)
        
        # Campaign response features
        df['campaign_response_rate'] = df['avg_ctr'] * df['avg_conversion_rate']
        df['ab_test_impact'] = df['avg_ctr_lift'] * df['avg_conversion_lift']
        
        # Risk and opportunity features
        df['churn_risk'] = 1 / (df['recency_days'] + 1)
        df['upsell_potential'] = df['monetary_value'] * df['purchase_velocity']
        df['lifetime_value_potential'] = df['customer_lifetime_value'] * df['engagement_score_normalized']
        
        # Categorical encoding
        df['is_high_value'] = (df['customer_value_tier'] == 'Premium').astype(int)
        df['is_engaged'] = (df['engagement_level'].isin(['Highly Engaged', 'Engaged'])).astype(int)
        df['is_mobile_user'] = (df['device_preference'] == 'Mobile').astype(int)
        df['is_return_visitor'] = (df['total_sessions'] > 1).astype(int)
        
        logger.info("✅ Feature engineering completed")
        return df
    
    def clean_unified_dataset(self, df):
        """Clean and finalize the unified dataset"""
        logger.info("🧹 Cleaning unified dataset...")
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna('Unknown')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['customer_id'])
        
        # Sort by customer_id for consistency
        df = df.sort_values('customer_id').reset_index(drop=True)
        
        # Add metadata
        df['dataset_created_at'] = datetime.now()
        df['data_version'] = '1.0'
        
        logger.info("✅ Dataset cleaning completed")
        return df
    
    def save_unified_dataset(self, df):
        """Save the unified dataset to parquet format"""
        logger.info("💾 Saving unified dataset...")
        
        # Create output directory
        output_path = 'data_pipelines/unified_dataset/output'
        os.makedirs(output_path, exist_ok=True)
        
        # Save main dataset
        output_file = os.path.join(output_path, 'unified_customer_dataset.parquet')
        df.to_parquet(output_file, index=False)
        
        # Save feature summary
        feature_summary = {
            'total_customers': len(df),
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'created_at': datetime.now().isoformat(),
            'feature_list': list(df.columns)
        }
        
        summary_file = os.path.join(output_path, 'unified_dataset_summary.yaml')
        with open(summary_file, 'w') as f:
            yaml.dump(feature_summary, f, default_flow_style=False)
        
        logger.info(f"✅ Unified dataset saved to {output_file}")
        logger.info(f"✅ Feature summary saved to {summary_file}")
        
        return output_file
    
    def generate_dataset_report(self, df):
        """Generate a comprehensive dataset report"""
        logger.info("📊 Generating dataset report...")
        
        report = {
            'dataset_info': {
                'total_customers': len(df),
                'total_features': len(df.columns),
                'creation_date': datetime.now().isoformat()
            },
            'feature_categories': {
                'rfm_features': [col for col in df.columns if 'rfm' in col.lower() or col in ['recency_days', 'frequency', 'monetary_value']],
                'demographic_features': [col for col in df.columns if col in ['age_group', 'education_level', 'occupation', 'household_size', 'marital_status', 'interests', 'lifestyle_segment']],
                'journey_features': [col for col in df.columns if any(x in col.lower() for x in ['session', 'page', 'engagement', 'conversion', 'search', 'product'])],
                'campaign_features': [col for col in df.columns if any(x in col.lower() for x in ['campaign', 'ctr', 'roas', 'conversion', 'ab_test'])],
                'engineered_features': [col for col in df.columns if any(x in col.lower() for x in ['velocity', 'intensity', 'efficiency', 'potential', 'risk', 'ratio'])]
            },
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'duplicates': df.duplicated().sum(),
                'data_types': df.dtypes.to_dict()
            },
            'statistical_summary': {
                'numeric_summary': df.describe().to_dict(),
                'categorical_summary': {col: df[col].value_counts().to_dict() for col in df.select_dtypes(include=['object']).columns}
            }
        }
        
        # Save report
        report_file = os.path.join('data_pipelines/unified_dataset/output', 'unified_dataset_report.yaml')
        with open(report_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f"✅ Dataset report saved to {report_file}")
        return report
    
    def run_pipeline(self):
        """Run the complete unified dataset creation pipeline"""
        logger.info("🚀 Starting unified dataset creation pipeline...")
        
        try:
            # Connect to Trino
            self.connect_trino()
            
            # Create unified dataset
            df_unified = self.create_unified_dataset()
            
            # Save dataset
            output_file = self.save_unified_dataset(df_unified)
            
            # Generate report
            report = self.generate_dataset_report(df_unified)
            
            logger.info("=" * 60)
            logger.info("🎉 UNIFIED DATASET CREATION COMPLETED!")
            logger.info("=" * 60)
            logger.info(f"📊 Dataset: {len(df_unified)} customers, {len(df_unified.columns)} features")
            logger.info(f"💾 Saved to: {output_file}")
            logger.info(f"📈 Ready for ML pipeline consumption")
            
            return df_unified, output_file
            
        except Exception as e:
            logger.error(f"❌ Error in unified dataset creation: {e}")
            raise
        finally:
            # Close connections
            if self.trino_engine:
                self.trino_engine.dispose()

def main():
    """Main function"""
    try:
        creator = UnifiedDatasetCreator()
        df_unified, output_file = creator.run_pipeline()
        
        print(f"\n🎉 Unified dataset created successfully!")
        print(f"📁 Location: {output_file}")
        print(f"📊 Shape: {df_unified.shape}")
        print(f"🔧 Ready for ML pipeline training!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
