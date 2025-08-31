#!/usr/bin/env python3
"""
MongoDB Data Generator for AMATO Production
Generates clickstream data for MongoDB database
"""

import pymongo
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

class MongoDBDataGenerator:
    def __init__(self, config_path='config/database_config.yaml'):
        self.config = self.load_config(config_path)
        self.fake = Faker()
        self.client = None
        self.db = None
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def connect_mongodb(self):
        """Connect to MongoDB database"""
        mongo_config = self.config['databases']['mongodb']
        connection_string = f"mongodb://{mongo_config['username']}:{mongo_config['password']}@{mongo_config['host']}:{mongo_config['port']}/{mongo_config['database']}?authSource={mongo_config['auth_source']}"
        
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[mongo_config['database']]
        logger.info(f"Connected to MongoDB database: {mongo_config['database']}")
    
    def close_mongodb(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def generate_sessions(self, count=100000):
        """Generate session data"""
        logger.info(f"Generating {count} sessions...")
        
        sessions = []
        device_types = ['Desktop', 'Mobile', 'Tablet']
        browsers = ['Chrome', 'Firefox', 'Safari', 'Edge', 'Opera']
        operating_systems = ['Windows', 'macOS', 'Linux', 'iOS', 'Android']
        
        for i in range(count):
            session_start = self.fake.date_time_between(start_date='-6m', end_date='now')
            session_end = session_start + timedelta(minutes=random.randint(1, 120))
            duration_seconds = int((session_end - session_start).total_seconds())
            
            session = {
                'session_id': f"SESS_{str(i+1).zfill(8)}",
                'customer_id': f"CUST_{random.randint(1, 10000):06d}",
                'session_start': session_start,
                'session_end': session_end,
                'duration_seconds': duration_seconds,
                'device_type': random.choice(device_types),
                'browser': random.choice(browsers),
                'os': random.choice(operating_systems),
                'ip_address': self.fake.ipv4(),
                'user_agent': self.fake.user_agent(),
                'referrer_url': random.choice([self.fake.url(), 'Direct', 'Google', 'Facebook', 'Twitter']),
                'landing_page': random.choice(['/home', '/products', '/category/electronics', '/category/clothing', '/deals']),
                'exit_page': random.choice(['/checkout', '/products', '/category/electronics', '/category/clothing', '/deals']),
                'pages_viewed': random.randint(1, 20),
                'created_at': datetime.now()
            }
            sessions.append(session)
        
        # Insert into database
        self.db.sessions.insert_many(sessions)
        logger.info(f"✅ Generated {len(sessions)} sessions")
        return sessions
    
    def generate_page_views(self, sessions, count=800000):
        """Generate page view data"""
        logger.info(f"Generating {count} page views...")
        
        page_views = []
        pages = [
            {'url': '/home', 'title': 'Homepage', 'category': 'Home', 'type': 'Landing'},
            {'url': '/products', 'title': 'Products', 'category': 'Products', 'type': 'Category'},
            {'url': '/category/electronics', 'title': 'Electronics', 'category': 'Electronics', 'type': 'Category'},
            {'url': '/category/clothing', 'title': 'Clothing', 'category': 'Clothing', 'type': 'Category'},
            {'url': '/category/books', 'title': 'Books', 'category': 'Books', 'type': 'Category'},
            {'url': '/product/laptop', 'title': 'Laptop Product', 'category': 'Electronics', 'type': 'Product'},
            {'url': '/product/smartphone', 'title': 'Smartphone Product', 'category': 'Electronics', 'type': 'Product'},
            {'url': '/cart', 'title': 'Shopping Cart', 'category': 'Cart', 'type': 'Cart'},
            {'url': '/checkout', 'title': 'Checkout', 'category': 'Checkout', 'type': 'Checkout'},
            {'url': '/deals', 'title': 'Deals', 'category': 'Deals', 'type': 'Promotional'}
        ]
        
        for i in range(count):
            session = random.choice(sessions)
            page = random.choice(pages)
            timestamp = self.fake.date_time_between(start_date=session['session_start'], end_date=session['session_end'])
            
            page_view = {
                'view_id': f"VIEW_{str(i+1).zfill(8)}",
                'session_id': session['session_id'],
                'customer_id': session['customer_id'],
                'page_url': page['url'],
                'page_title': page['title'],
                'page_category': page['category'],
                'page_type': page['type'],
                'time_on_page': random.randint(10, 600),  # 10 seconds to 10 minutes
                'scroll_depth': random.randint(0, 100),  # 0% to 100%
                'timestamp': timestamp,
                'created_at': datetime.now()
            }
            page_views.append(page_view)
        
        # Insert into database
        self.db.page_views.insert_many(page_views)
        logger.info(f"✅ Generated {len(page_views)} page views")
        return page_views
    
    def generate_events(self, sessions, count=600000):
        """Generate event data"""
        logger.info(f"Generating {count} events...")
        
        events = []
        event_types = ['click', 'form_submit', 'scroll', 'hover', 'add_to_cart', 'remove_from_cart', 'search']
        event_names = {
            'click': ['button_click', 'link_click', 'image_click'],
            'form_submit': ['newsletter_signup', 'contact_form', 'login_form'],
            'scroll': ['page_scroll', 'section_scroll'],
            'hover': ['product_hover', 'menu_hover'],
            'add_to_cart': ['add_to_cart'],
            'remove_from_cart': ['remove_from_cart'],
            'search': ['search_query', 'search_filter']
        }
        
        for i in range(count):
            session = random.choice(sessions)
            event_type = random.choice(event_types)
            event_name = random.choice(event_names[event_type])
            timestamp = self.fake.date_time_between(start_date=session['session_start'], end_date=session['session_end'])
            
            # Generate event-specific data
            event_data = {}
            if event_type == 'click':
                event_data = {'element_id': f'btn_{random.randint(1, 100)}', 'element_text': 'Click me'}
            elif event_type == 'form_submit':
                event_data = {'form_id': f'form_{random.randint(1, 10)}', 'fields_count': random.randint(1, 5)}
            elif event_type == 'add_to_cart':
                event_data = {'product_id': f'PROD_{random.randint(1, 1000)}', 'quantity': random.randint(1, 5)}
            elif event_type == 'search':
                event_data = {'query': self.fake.word(), 'results_count': random.randint(0, 100)}
            
            event = {
                'event_id': f"EVENT_{str(i+1).zfill(8)}",
                'session_id': session['session_id'],
                'customer_id': session['customer_id'],
                'event_type': event_type,
                'event_name': event_name,
                'event_data': event_data,
                'timestamp': timestamp,
                'created_at': datetime.now()
            }
            events.append(event)
        
        # Insert into database
        self.db.events.insert_many(events)
        logger.info(f"✅ Generated {len(events)} events")
        return events
    
    def generate_product_interactions(self, sessions, count=400000):
        """Generate product interaction data"""
        logger.info(f"Generating {count} product interactions...")
        
        product_interactions = []
        interaction_types = ['view', 'add_to_wishlist', 'add_to_cart', 'remove_from_cart', 'review', 'share']
        
        for i in range(count):
            session = random.choice(sessions)
            interaction_type = random.choice(interaction_types)
            timestamp = self.fake.date_time_between(start_date=session['session_start'], end_date=session['session_end'])
            
            interaction_data = {}
            if interaction_type == 'view':
                interaction_data = {'view_duration': random.randint(5, 300)}
            elif interaction_type in ['add_to_cart', 'remove_from_cart']:
                interaction_data = {'quantity': random.randint(1, 5)}
            elif interaction_type == 'review':
                interaction_data = {'rating': random.randint(1, 5), 'review_text': self.fake.text(max_nb_chars=200)}
            elif interaction_type == 'share':
                interaction_data = {'platform': random.choice(['Facebook', 'Twitter', 'Email', 'WhatsApp'])}
            
            interaction = {
                'interaction_id': f"INT_{str(i+1).zfill(8)}",
                'session_id': session['session_id'],
                'customer_id': session['customer_id'],
                'product_id': f"PROD_{random.randint(1, 1000)}",
                'interaction_type': interaction_type,
                'interaction_data': interaction_data,
                'timestamp': timestamp,
                'created_at': datetime.now()
            }
            product_interactions.append(interaction)
        
        # Insert into database
        self.db.product_interactions.insert_many(product_interactions)
        logger.info(f"✅ Generated {len(product_interactions)} product interactions")
        return product_interactions
    
    def generate_search_queries(self, sessions, count=300000):
        """Generate search query data"""
        logger.info(f"Generating {count} search queries...")
        
        search_queries = []
        search_terms = [
            'laptop', 'smartphone', 'headphones', 'shoes', 'books', 'coffee', 'watch',
            'backpack', 't-shirt', 'jeans', 'camera', 'tablet', 'keyboard', 'mouse',
            'speakers', 'gaming', 'fitness', 'kitchen', 'home', 'office', 'travel'
        ]
        
        for i in range(count):
            session = random.choice(sessions)
            search_term = random.choice(search_terms)
            timestamp = self.fake.date_time_between(start_date=session['session_start'], end_date=session['session_end'])
            
            search_query = {
                'query_id': f"QUERY_{str(i+1).zfill(8)}",
                'session_id': session['session_id'],
                'customer_id': session['customer_id'],
                'search_term': search_term,
                'search_results_count': random.randint(0, 100),
                'clicked_result_position': random.randint(1, 10) if random.random() > 0.3 else None,
                'timestamp': timestamp,
                'created_at': datetime.now()
            }
            search_queries.append(search_query)
        
        # Insert into database
        self.db.search_queries.insert_many(search_queries)
        logger.info(f"✅ Generated {len(search_queries)} search queries")
        return search_queries
    
    def generate_all_data(self):
        """Generate all MongoDB data"""
        logger.info("🚀 Starting MongoDB data generation...")
        
        try:
            self.connect_mongodb()
            
            # Generate sessions
            sessions = self.generate_sessions(self.config['data_generation']['sessions_count'])
            
            # Generate page views
            self.generate_page_views(sessions, self.config['data_generation']['page_views_count'])
            
            # Generate events
            self.generate_events(sessions, self.config['data_generation']['events_count'])
            
            # Generate product interactions
            self.generate_product_interactions(sessions, self.config['data_generation']['product_interactions_count'])
            
            # Generate search queries
            self.generate_search_queries(sessions, self.config['data_generation']['search_queries_count'])
            
            logger.info("✅ MongoDB data generation completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in MongoDB data generation: {e}")
            raise
        finally:
            self.close_mongodb()

if __name__ == "__main__":
    generator = MongoDBDataGenerator()
    generator.generate_all_data()
