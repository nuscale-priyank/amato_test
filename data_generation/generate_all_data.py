#!/usr/bin/env python3
"""
AMATO Production Data Generation Orchestrator
Generates data for all databases (MySQL, PostgreSQL, MongoDB)
"""

import sys
import os
import logging
import yaml
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generation.mysql_data_generator import MySQLDataGenerator
from data_generation.postgresql_data_generator import PostgreSQLDataGenerator
from data_generation.mongodb_data_generator import MongoDBDataGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataGenerationOrchestrator:
    def __init__(self, config_path='config/database_config.yaml'):
        self.config = self.load_config(config_path)
        self.start_time = datetime.now()
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def create_logs_directory(self):
        """Create logs directory if it doesn't exist"""
        os.makedirs('logs', exist_ok=True)
    
    def generate_mysql_data(self):
        """Generate MySQL data"""
        logger.info("🔄 Starting MySQL data generation...")
        try:
            mysql_generator = MySQLDataGenerator()
            mysql_generator.generate_all_data()
            logger.info("✅ MySQL data generation completed successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ MySQL data generation failed: {e}")
            return False
    
    def generate_postgresql_data(self):
        """Generate PostgreSQL data"""
        logger.info("🔄 Starting PostgreSQL data generation...")
        try:
            pg_generator = PostgreSQLDataGenerator()
            pg_generator.generate_all_data()
            logger.info("✅ PostgreSQL data generation completed successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ PostgreSQL data generation failed: {e}")
            return False
    
    def generate_mongodb_data(self):
        """Generate MongoDB data"""
        logger.info("🔄 Starting MongoDB data generation...")
        try:
            mongo_generator = MongoDBDataGenerator()
            mongo_generator.generate_all_data()
            logger.info("✅ MongoDB data generation completed successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ MongoDB data generation failed: {e}")
            return False
    
    def generate_all_data(self):
        """Generate data for all databases"""
        logger.info("🚀 Starting AMATO Production Data Generation...")
        logger.info(f"Start time: {self.start_time}")
        
        # Create logs directory
        self.create_logs_directory()
        
        # Track success/failure
        results = {
            'mysql': False,
            'postgresql': False,
            'mongodb': False
        }
        
        # Generate MySQL data
        results['mysql'] = self.generate_mysql_data()
        
        # Generate PostgreSQL data
        results['postgresql'] = self.generate_postgresql_data()
        
        # Generate MongoDB data
        results['mongodb'] = self.generate_mongodb_data()
        
        # Summary
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        logger.info("=" * 60)
        logger.info("📊 DATA GENERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Start time: {self.start_time}")
        logger.info(f"End time: {end_time}")
        logger.info(f"Duration: {duration}")
        logger.info("")
        logger.info("Database Results:")
        for db, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            logger.info(f"  {db.upper()}: {status}")
        
        # Overall status
        all_success = all(results.values())
        if all_success:
            logger.info("")
            logger.info("🎉 ALL DATA GENERATION COMPLETED SUCCESSFULLY!")
            logger.info("")
            logger.info("📈 Generated Data Summary:")
            logger.info(f"  MySQL: {self.config['data_generation']['customers_count']} customers, {self.config['data_generation']['transactions_count']} transactions")
            logger.info(f"  PostgreSQL: {self.config['data_generation']['campaigns_count']} campaigns, {self.config['data_generation']['ab_tests_count']} A/B tests")
            logger.info(f"  MongoDB: {self.config['data_generation']['sessions_count']} sessions, {self.config['data_generation']['page_views_count']} page views")
        else:
            logger.error("")
            logger.error("💥 SOME DATA GENERATION FAILED!")
            logger.error("Please check the logs above for details.")
        
        return all_success

def main():
    """Main function"""
    try:
        orchestrator = DataGenerationOrchestrator()
        success = orchestrator.generate_all_data()
        
        if success:
            print("\n🎉 Data generation completed successfully!")
            print("You can now proceed with the data pipelines.")
        else:
            print("\n💥 Data generation failed!")
            print("Please check the logs for details.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error in data generation: {e}")
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
