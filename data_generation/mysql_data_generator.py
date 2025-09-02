#!/usr/bin/env python3
"""
MySQL Data Generator for AMATO Production
Generates customer and transaction data for MySQL database
"""

import mysql.connector
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

class MySQLDataGenerator:
    def __init__(self, config_path='config/database_config.yaml'):
        self.config = self.load_config(config_path)
        self.fake = Faker()
        self.conn = None
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def connect_mysql(self):
        """Connect to MySQL database"""
        mysql_config = self.config['databases']['mysql']
        self.conn = mysql.connector.connect(
            host=mysql_config['host'],
            port=mysql_config['port'],
            database=mysql_config['database'],
            user=mysql_config['username'],
            password=mysql_config['password'],
            charset=mysql_config['charset']
        )
        logger.info(f"Connected to MySQL database: {mysql_config['database']}")
    
    def close_mysql(self):
        """Close MySQL connection"""
        if self.conn:
            self.conn.close()
            logger.info("MySQL connection closed")
    
    def clear_existing_data(self):
        """Clear existing data from all tables"""
        logger.info("🧹 Clearing existing data from MySQL tables...")
        cursor = self.conn.cursor()
        
        try:
            # Clear tables in reverse dependency order
            tables_to_clear = [
                'transactions', 'customer_segments', 'customers'
            ]
            
            for table in tables_to_clear:
                cursor.execute(f"DELETE FROM {table}")
                logger.info(f"✅ Cleared table: {table}")
            
            # Reset auto-increment counters
            for table in tables_to_clear:
                cursor.execute(f"ALTER TABLE {table} AUTO_INCREMENT = 1")
                logger.info(f"✅ Reset auto-increment for: {table}")
            
            self.conn.commit()
            logger.info("✅ All existing data cleared from MySQL")
            
        except Exception as e:
            logger.error(f"❌ Error clearing data: {e}")
            self.conn.rollback()
            raise
        finally:
            cursor.close()
    
    def generate_customers(self, count=10000):
        """Generate customer data"""
        logger.info(f"Generating {count} customers...")
        
        customers = []
        used_emails = set()
        
        for i in range(count):
            customer_id = f"CUST_{str(i+1).zfill(6)}"
            registration_date = self.fake.date_between(start_date='-2y', end_date='today')
            
            # Generate unique email
            email = self.fake.email()
            while email in used_emails:
                email = self.fake.email()
            used_emails.add(email)
            
            customer = {
                'customer_id': customer_id,
                'email': email,
                'first_name': self.fake.first_name(),
                'last_name': self.fake.last_name(),
                'date_of_birth': self.fake.date_of_birth(minimum_age=18, maximum_age=80),
                'gender': random.choice(['M', 'F', 'O']),
                'city': self.fake.city(),
                'state': self.fake.state(),
                'country': self.fake.country(),
                'postal_code': self.fake.postcode(),
                'income_bracket': random.choice(['Low', 'Medium', 'High', 'Very High']),
                'registration_date': registration_date,
                'last_login_date': self.fake.date_between(start_date=registration_date, end_date='today'),
                'is_active': random.choice([True, True, True, False]),  # 75% active
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            customers.append(customer)
        
        # Insert into database
        cursor = self.conn.cursor()
        insert_query = """
        INSERT INTO customers (
            customer_id, email, first_name, last_name, date_of_birth, gender,
            city, state, country, postal_code, income_bracket, registration_date,
            last_login_date, is_active, created_at, updated_at
        ) VALUES (
            %(customer_id)s, %(email)s, %(first_name)s, %(last_name)s, %(date_of_birth)s, %(gender)s,
            %(city)s, %(state)s, %(country)s, %(postal_code)s, %(income_bracket)s, %(registration_date)s,
            %(last_login_date)s, %(is_active)s, %(created_at)s, %(updated_at)s
        )
        """
        
        cursor.executemany(insert_query, customers)
        self.conn.commit()
        cursor.close()
        
        logger.info(f"✅ Generated {len(customers)} customers")
        return customers
    
    def generate_customer_demographics(self, customers):
        """Generate customer demographics data"""
        logger.info(f"Generating demographics for {len(customers)} customers...")
        
        demographics = []
        for customer in customers:
            age = (datetime.now().date() - customer['date_of_birth']).days // 365
            
            if age < 25:
                age_group = "18-24"
            elif age < 35:
                age_group = "25-34"
            elif age < 45:
                age_group = "35-44"
            elif age < 55:
                age_group = "45-54"
            else:
                age_group = "55+"
            
            demographic = {
                'customer_id': customer['customer_id'],
                'age_group': age_group,
                'education_level': random.choice(['High School', 'Bachelor', 'Master', 'PhD', 'Other']),
                'occupation': random.choice(['Engineer', 'Manager', 'Sales', 'Marketing', 'Student', 'Retired', 'Other']),
                'household_size': random.randint(1, 6),
                'marital_status': random.choice(['Single', 'Married', 'Divorced', 'Widowed']),
                'interests': random.choice(['Technology', 'Sports', 'Travel', 'Food', 'Fashion', 'Music', 'Books']),
                'lifestyle_segment': random.choice(['Urban', 'Suburban', 'Rural']),
                'created_at': datetime.now()
            }
            demographics.append(demographic)
        
        # Insert into database
        cursor = self.conn.cursor()
        insert_query = """
        INSERT INTO customer_demographics (
            customer_id, age_group, education_level, occupation, household_size,
            marital_status, interests, lifestyle_segment, created_at
        ) VALUES (
            %(customer_id)s, %(age_group)s, %(education_level)s, %(occupation)s, %(household_size)s,
            %(marital_status)s, %(interests)s, %(lifestyle_segment)s, %(created_at)s
        )
        """
        
        cursor.executemany(insert_query, demographics)
        self.conn.commit()
        cursor.close()
        
        logger.info(f"✅ Generated demographics for {len(demographics)} customers")
        return demographics
    
    def generate_transactions(self, customers, count=50000):
        """Generate transaction data"""
        logger.info(f"Generating {count} transactions...")
        
        transactions = []
        for i in range(count):
            customer = random.choice(customers)
            order_date = self.fake.date_between(start_date=customer['registration_date'], end_date='today')
            
            transaction = {
                'transaction_id': f"TXN_{str(i+1).zfill(8)}",
                'customer_id': customer['customer_id'],
                'order_date': order_date,
                'total_amount': round(random.uniform(10.0, 1000.0), 2),
                'currency': 'USD',
                'payment_method': random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer']),
                'shipping_address': self.fake.address(),
                'order_status': random.choice(['Delivered', 'Delivered', 'Delivered', 'Shipped', 'Processing', 'Cancelled']),
                'created_at': datetime.now()
            }
            transactions.append(transaction)
        
        # Insert into database
        cursor = self.conn.cursor()
        insert_query = """
        INSERT INTO transactions (
            transaction_id, customer_id, order_date, total_amount, currency,
            payment_method, shipping_address, order_status, created_at
        ) VALUES (
            %(transaction_id)s, %(customer_id)s, %(order_date)s, %(total_amount)s, %(currency)s,
            %(payment_method)s, %(shipping_address)s, %(order_status)s, %(created_at)s
        )
        """
        
        cursor.executemany(insert_query, transactions)
        self.conn.commit()
        cursor.close()
        
        logger.info(f"✅ Generated {len(transactions)} transactions")
        return transactions
    
    def generate_transaction_items(self, transactions, count=150000):
        """Generate transaction items data"""
        logger.info(f"Generating {count} transaction items...")
        
        products = [
            {'name': 'Laptop', 'category': 'Electronics', 'subcategory': 'Computers'},
            {'name': 'Smartphone', 'category': 'Electronics', 'subcategory': 'Mobile'},
            {'name': 'Headphones', 'category': 'Electronics', 'subcategory': 'Audio'},
            {'name': 'T-Shirt', 'category': 'Clothing', 'subcategory': 'Casual'},
            {'name': 'Jeans', 'category': 'Clothing', 'subcategory': 'Casual'},
            {'name': 'Sneakers', 'category': 'Footwear', 'subcategory': 'Casual'},
            {'name': 'Book', 'category': 'Books', 'subcategory': 'Fiction'},
            {'name': 'Coffee Mug', 'category': 'Home', 'subcategory': 'Kitchen'},
            {'name': 'Backpack', 'category': 'Accessories', 'subcategory': 'Bags'},
            {'name': 'Watch', 'category': 'Accessories', 'subcategory': 'Jewelry'}
        ]
        
        transaction_items = []
        for i in range(count):
            transaction = random.choice(transactions)
            product = random.choice(products)
            quantity = random.randint(1, 5)
            unit_price = round(random.uniform(5.0, 500.0), 2)
            
            item = {
                'item_id': f"ITEM_{str(i+1).zfill(8)}",
                'transaction_id': transaction['transaction_id'],
                'product_id': f"PROD_{random.randint(1, 1000)}",
                'product_name': product['name'],
                'category': product['category'],
                'subcategory': product['subcategory'],
                'quantity': quantity,
                'unit_price': unit_price,
                'total_price': round(unit_price * quantity, 2),
                'created_at': datetime.now()
            }
            transaction_items.append(item)
        
        # Insert into database
        cursor = self.conn.cursor()
        insert_query = """
        INSERT INTO transaction_items (
            item_id, transaction_id, product_id, product_name, category,
            subcategory, quantity, unit_price, total_price, created_at
        ) VALUES (
            %(item_id)s, %(transaction_id)s, %(product_id)s, %(product_name)s, %(category)s,
            %(subcategory)s, %(quantity)s, %(unit_price)s, %(total_price)s, %(created_at)s
        )
        """
        
        cursor.executemany(insert_query, transaction_items)
        self.conn.commit()
        cursor.close()
        
        logger.info(f"✅ Generated {len(transaction_items)} transaction items")
        return transaction_items
    
    def generate_all_data(self):
        """Generate all MySQL data"""
        logger.info("🚀 Starting MySQL data generation...")
        
        try:
            self.connect_mysql()
            
            # Clear existing data first
            self.clear_existing_data()
            
            # Generate customers
            customers = self.generate_customers(self.config['data_generation']['customers_count'])
            
            # Generate customer demographics
            self.generate_customer_demographics(customers)
            
            # Generate transactions
            transactions = self.generate_transactions(customers, self.config['data_generation']['transactions_count'])
            
            # Generate transaction items
            self.generate_transaction_items(transactions, self.config['data_generation']['transactions_count'] * 3)
            
            logger.info("✅ MySQL data generation completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in MySQL data generation: {e}")
            raise
        finally:
            self.close_mysql()

if __name__ == "__main__":
    generator = MySQLDataGenerator()
    generator.generate_all_data()
