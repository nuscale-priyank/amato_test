-- MySQL Schema for AMATO Production
-- Customer and Transaction Data

-- Create database
CREATE DATABASE IF NOT EXISTS amato;
USE amato;

-- Customer tables
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    date_of_birth DATE,
    gender ENUM('M', 'F', 'O'),
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100),
    postal_code VARCHAR(20),
    income_bracket VARCHAR(50),
    registration_date DATETIME,
    last_login_date DATETIME,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_email (email),
    INDEX idx_registration_date (registration_date),
    INDEX idx_is_active (is_active)
);

CREATE TABLE customer_demographics (
    customer_id VARCHAR(50) PRIMARY KEY,
    age_group VARCHAR(20),
    education_level VARCHAR(50),
    occupation VARCHAR(100),
    household_size INT,
    marital_status VARCHAR(20),
    interests TEXT,
    lifestyle_segment VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    INDEX idx_age_group (age_group),
    INDEX idx_education_level (education_level),
    INDEX idx_lifestyle_segment (lifestyle_segment)
);

-- Transaction tables
CREATE TABLE transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    order_date DATETIME,
    total_amount DECIMAL(10,2),
    currency VARCHAR(3) DEFAULT 'USD',
    payment_method VARCHAR(50),
    shipping_address TEXT,
    order_status ENUM('Pending', 'Processing', 'Shipped', 'Delivered', 'Cancelled', 'Refunded'),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL,
    INDEX idx_customer_id (customer_id),
    INDEX idx_order_date (order_date),
    INDEX idx_order_status (order_status),
    INDEX idx_total_amount (total_amount)
);

CREATE TABLE transaction_items (
    item_id VARCHAR(50) PRIMARY KEY,
    transaction_id VARCHAR(50),
    product_id VARCHAR(50),
    product_name VARCHAR(255),
    category VARCHAR(100),
    subcategory VARCHAR(100),
    quantity INT,
    unit_price DECIMAL(10,2),
    total_price DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    INDEX idx_transaction_id (transaction_id),
    INDEX idx_product_id (product_id),
    INDEX idx_category (category)
);

-- Customer segmentation tables
CREATE TABLE customer_segments (
    segment_id VARCHAR(50) PRIMARY KEY,
    segment_name VARCHAR(100) UNIQUE NOT NULL,
    segment_description TEXT,
    criteria JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_segment_name (segment_name)
);

CREATE TABLE customer_segment_mapping (
    mapping_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    segment_id VARCHAR(50),
    assigned_date DATETIME,
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    FOREIGN KEY (segment_id) REFERENCES customer_segments(segment_id) ON DELETE CASCADE,
    UNIQUE KEY unique_customer_segment (customer_id, segment_id),
    INDEX idx_customer_id (customer_id),
    INDEX idx_segment_id (segment_id),
    INDEX idx_assigned_date (assigned_date)
);

-- Insert sample segments
INSERT INTO customer_segments (segment_id, segment_name, segment_description, criteria) VALUES
('SEG_001', 'High Value Customers', 'Customers with high RFM scores and significant purchase history', '{"rfm_score": "5", "monetary_value": ">1000"}'),
('SEG_002', 'Loyal Customers', 'Customers with high frequency and recent purchases', '{"frequency": ">10", "recency_days": "<30"}'),
('SEG_003', 'At Risk Customers', 'Customers with declining engagement or no recent purchases', '{"recency_days": ">90", "frequency": "<3"}'),
('SEG_004', 'New Customers', 'Recently acquired customers with potential for growth', '{"registration_date": ">30 days ago", "frequency": "<5"}'),
('SEG_005', 'Churned Customers', 'Customers who have stopped engaging', '{"recency_days": ">180", "is_active": false}');
