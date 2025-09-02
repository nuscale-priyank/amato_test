#!/usr/bin/env python3
"""
PostgreSQL Data Generator for AMATO Production
Generates campaign and A/B test data for PostgreSQL database
"""

import psycopg2
import pandas as pd
import numpy as np
from faker import Faker
import yaml
import logging
import random
from datetime import datetime, timedelta
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgreSQLDataGenerator:
    def __init__(self, config_path='config/database_config.yaml'):
        self.config = self.load_config(config_path)
        self.fake = Faker()
        self.conn = None
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def connect_postgresql(self):
        """Connect to PostgreSQL database"""
        pg_config = self.config['databases']['postgresql']
        self.conn = psycopg2.connect(
            host=pg_config['host'],
            port=pg_config['port'],
            database=pg_config['database'],
            user=pg_config['username'],
            password=pg_config['password']
        )
        # Set the schema
        cursor = self.conn.cursor()
        cursor.execute(f"SET search_path TO {pg_config['schema']}")
        self.conn.commit()
        cursor.close()
        logger.info(f"Connected to PostgreSQL database: {pg_config['database']} with schema: {pg_config['schema']}")
    
    def close_postgresql(self):
        """Close PostgreSQL connection"""
        if self.conn:
            self.conn.close()
            logger.info("PostgreSQL connection closed")
    
    def clear_existing_data(self):
        """Clear existing data from all tables"""
        logger.info("🧹 Clearing existing data from PostgreSQL tables...")
        cursor = self.conn.cursor()
        
        try:
            # Clear tables in reverse dependency order
            tables_to_clear = [
                'ab_test_results', 'ab_tests', 'campaign_performance', 'campaigns'
            ]
            
            for table in tables_to_clear:
                cursor.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
                logger.info(f"✅ Cleared table: {table}")
            
            # TRUNCATE with RESTART IDENTITY automatically resets any auto-incrementing columns
            
            self.conn.commit()
            logger.info("✅ All existing data cleared from PostgreSQL")
            
        except Exception as e:
            logger.error(f"❌ Error clearing data: {e}")
            self.conn.rollback()
            raise
        finally:
            cursor.close()
    
    def generate_campaigns(self, count=100):
        """Generate campaign data"""
        logger.info(f"Generating {count} campaigns...")
        
        campaigns = []
        campaign_types = ['Email', 'Social Media', 'Search', 'Display', 'Video', 'Affiliate']
        channels = ['Facebook', 'Google', 'Instagram', 'LinkedIn', 'Twitter', 'Email', 'Direct']
        target_audiences = ['New Customers', 'Existing Customers', 'High Value', 'At Risk', 'Loyal', 'All']
        
        for i in range(count):
            start_date = self.fake.date_between(start_date='-1y', end_date='today')
            end_date = self.fake.date_between(start_date=start_date, end_date='+6m')
            
            campaign = {
                'campaign_id': f"CAMP_{str(i+1).zfill(6)}",
                'campaign_name': f"{random.choice(campaign_types)} Campaign {i+1}",
                'campaign_type': random.choice(campaign_types),
                'channel': random.choice(channels),
                'target_audience': random.choice(target_audiences),
                'start_date': start_date,
                'end_date': end_date,
                'budget': round(random.uniform(1000.0, 50000.0), 2),
                'status': random.choice(['Active', 'Active', 'Active', 'Paused', 'Completed']),
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            campaigns.append(campaign)
        
        # Insert into database
        cursor = self.conn.cursor()
        # Set schema explicitly
        cursor.execute(f"SET search_path TO {self.config['databases']['postgresql']['schema']}")
        
        insert_query = """
        INSERT INTO campaigns (
            campaign_id, campaign_name, campaign_type, channel, target_audience,
            start_date, end_date, budget, status, created_at, updated_at
        ) VALUES (
            %(campaign_id)s, %(campaign_name)s, %(campaign_type)s, %(channel)s, %(target_audience)s,
            %(start_date)s, %(end_date)s, %(budget)s, %(status)s, %(created_at)s, %(updated_at)s
        )
        """
        
        cursor.executemany(insert_query, campaigns)
        self.conn.commit()
        cursor.close()
        
        logger.info(f"✅ Generated {len(campaigns)} campaigns")
        return campaigns
    
    def generate_campaign_performance(self, campaigns, count=956):
        """Generate campaign performance data"""
        logger.info(f"Generating {count} campaign performance records...")
        
        performance_records = []
        for i in range(count):
            campaign = random.choice(campaigns)
            date = self.fake.date_between(start_date=campaign['start_date'], end_date=campaign['end_date'])
            
            impressions = random.randint(1000, 100000)
            clicks = random.randint(50, int(impressions * 0.1))  # 0.1% to 10% CTR
            conversions = random.randint(1, int(clicks * 0.3))   # 1% to 30% conversion rate
            revenue = round(conversions * random.uniform(10.0, 200.0), 2)
            
            ctr = round(clicks / impressions, 4) if impressions > 0 else 0
            conversion_rate = round(conversions / clicks, 4) if clicks > 0 else 0
            roas = round(revenue / (campaign['budget'] / 365), 2) if campaign['budget'] > 0 else 0
            # Cap ROAS to fit in DECIMAL(5,2) field (max 999.99)
            roas = min(roas, 999.99)
            
            performance = {
                'performance_id': f"PERF_{str(i+1).zfill(8)}",
                'campaign_id': campaign['campaign_id'],
                'date': date,
                'impressions': impressions,
                'clicks': clicks,
                'conversions': conversions,
                'revenue': revenue,
                'ctr': ctr,
                'conversion_rate': conversion_rate,
                'roas': roas,
                'created_at': datetime.now()
            }
            performance_records.append(performance)
        
        # Insert into database
        cursor = self.conn.cursor()
        # Set schema explicitly
        cursor.execute(f"SET search_path TO {self.config['databases']['postgresql']['schema']}")
        
        insert_query = """
        INSERT INTO campaign_performance (
            performance_id, campaign_id, date, impressions, clicks, conversions,
            revenue, ctr, conversion_rate, roas, created_at
        ) VALUES (
            %(performance_id)s, %(campaign_id)s, %(date)s, %(impressions)s, %(clicks)s, %(conversions)s,
            %(revenue)s, %(ctr)s, %(conversion_rate)s, %(roas)s, %(created_at)s
        )
        """
        
        cursor.executemany(insert_query, performance_records)
        self.conn.commit()
        cursor.close()
        
        logger.info(f"✅ Generated {len(performance_records)} campaign performance records")
        return performance_records
    
    def generate_ab_tests(self, campaigns, count=50):
        """Generate A/B test data"""
        logger.info(f"Generating {count} A/B tests...")
        
        ab_tests = []
        test_types = ['Email Subject', 'Landing Page', 'Ad Copy', 'Button Color', 'Pricing', 'Layout']
        
        for i in range(count):
            campaign = random.choice(campaigns)
            start_date = self.fake.date_between(start_date=campaign['start_date'], end_date=campaign['end_date'])
            end_date = self.fake.date_between(start_date=start_date, end_date='+30d')
            
            test_type = random.choice(test_types)
            ab_test = {
                'test_id': f"ABT_{str(i+1).zfill(6)}",
                'test_name': f"{test_type} A/B Test {i+1}",
                'campaign_id': campaign['campaign_id'],
                'variant_a_description': f"Original {test_type.lower()}",
                'variant_b_description': f"New {test_type.lower()} design",
                'start_date': start_date,
                'end_date': end_date,
                'sample_size': random.randint(1000, 10000),
                'test_status': random.choice(['Running', 'Running', 'Completed', 'Paused']),
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            ab_tests.append(ab_test)
        
        # Insert into database
        cursor = self.conn.cursor()
        # Set schema explicitly
        cursor.execute(f"SET search_path TO {self.config['databases']['postgresql']['schema']}")
        
        insert_query = """
        INSERT INTO ab_tests (
            test_id, test_name, campaign_id, variant_a_description, variant_b_description,
            start_date, end_date, sample_size, test_status, created_at, updated_at
        ) VALUES (
            %(test_id)s, %(test_name)s, %(campaign_id)s, %(variant_a_description)s, %(variant_b_description)s,
            %(start_date)s, %(end_date)s, %(sample_size)s, %(test_status)s, %(created_at)s, %(updated_at)s
        )
        """
        
        cursor.executemany(insert_query, ab_tests)
        self.conn.commit()
        cursor.close()
        
        logger.info(f"✅ Generated {len(ab_tests)} A/B tests")
        return ab_tests
    
    def generate_ab_test_results(self, ab_tests, count=100):
        """Generate A/B test results data"""
        logger.info(f"Generating {count} A/B test results...")
        
        test_results = []
        for i in range(count):
            ab_test = random.choice(ab_tests)
            variant = random.choice(['A', 'B'])
            
            # Generate realistic test results
            base_impressions = random.randint(500, 5000)
            base_clicks = random.randint(50, int(base_impressions * 0.15))
            base_conversions = random.randint(5, int(base_clicks * 0.25))
            
            # Variant B might have slightly different performance
            if variant == 'B':
                impressions = int(base_impressions * random.uniform(0.8, 1.2))
                clicks = int(base_clicks * random.uniform(0.7, 1.3))
                conversions = int(base_conversions * random.uniform(0.6, 1.4))
            else:
                impressions = base_impressions
                clicks = base_clicks
                conversions = base_conversions
            
            revenue = round(conversions * random.uniform(10.0, 200.0), 2)
            ctr = round(clicks / impressions, 4) if impressions > 0 else 0
            conversion_rate = round(conversions / clicks, 4) if clicks > 0 else 0
            
            result = {
                'result_id': f"RES_{str(i+1).zfill(8)}",
                'test_id': ab_test['test_id'],
                'variant': variant,
                'impressions': impressions,
                'clicks': clicks,
                'conversions': conversions,
                'revenue': revenue,
                'ctr': ctr,
                'conversion_rate': conversion_rate,
                'created_at': datetime.now()
            }
            test_results.append(result)
        
        # Insert into database
        cursor = self.conn.cursor()
        # Set schema explicitly
        cursor.execute(f"SET search_path TO {self.config['databases']['postgresql']['schema']}")
        
        insert_query = """
        INSERT INTO ab_test_results (
            result_id, test_id, variant, impressions, clicks, conversions,
            revenue, ctr, conversion_rate, created_at
        ) VALUES (
            %(result_id)s, %(test_id)s, %(variant)s, %(impressions)s, %(clicks)s, %(conversions)s,
            %(revenue)s, %(ctr)s, %(conversion_rate)s, %(created_at)s
        )
        """
        
        cursor.executemany(insert_query, test_results)
        self.conn.commit()
        cursor.close()
        
        logger.info(f"✅ Generated {len(test_results)} A/B test results")
        return test_results
    
    def generate_all_data(self):
        """Generate all PostgreSQL data"""
        logger.info("🚀 Starting PostgreSQL data generation...")
        
        try:
            self.connect_postgresql()
            
            # Clear existing data first
            self.clear_existing_data()
            
            # Generate campaigns
            campaigns = self.generate_campaigns(self.config['data_generation']['campaigns_count'])
            
            # Generate campaign performance
            self.generate_campaign_performance(campaigns, self.config['data_generation']['campaigns_count'] * 10)
            
            # Generate A/B tests
            ab_tests = self.generate_ab_tests(campaigns, self.config['data_generation']['ab_tests_count'])
            
            # Generate A/B test results
            self.generate_ab_test_results(ab_tests, self.config['data_generation']['ab_tests_count'] * 2)
            
            logger.info("✅ PostgreSQL data generation completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in PostgreSQL data generation: {e}")
            raise
        finally:
            self.close_postgresql()

if __name__ == "__main__":
    generator = PostgreSQLDataGenerator()
    generator.generate_all_data()
