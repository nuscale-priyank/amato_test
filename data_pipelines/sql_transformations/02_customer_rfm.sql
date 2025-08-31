-- AMATO Production - Customer RFM Analysis SQL Script
-- This script calculates RFM (Recency, Frequency, Monetary) scores and segments customers
-- Run this script using Trino to analyze customer value and behavior

-- Set the session schema
USE nucentral_mysqldb.amato;

-- 1. Calculate RFM Metrics for each customer
CREATE TABLE IF NOT EXISTS nucentral_mysqldb.amato.customer_rfm_data AS
WITH customer_metrics AS (
    SELECT 
        c.customer_id,
        c.email,
        c.first_name,
        c.last_name,
        c.registration_date,
        
        -- Recency: Days since last purchase
        DATE_DIFF('day', MAX(t.order_date), CURRENT_DATE) as recency_days,
        
        -- Frequency: Number of orders
        COUNT(DISTINCT t.transaction_id) as frequency,
        
        -- Monetary: Total amount spent
        COALESCE(SUM(t.total_amount), 0) as monetary_value,
        
        -- Additional metrics
        AVG(t.total_amount) as avg_order_value,
        COUNT(DISTINCT ti.item_id) as total_items_purchased,
        COUNT(DISTINCT DATE(t.order_date)) as unique_purchase_days,
        
        -- Customer lifetime value (simplified calculation)
        COALESCE(SUM(t.total_amount), 0) * 1.25 as customer_lifetime_value,
        
        -- Purchase frequency rate (orders per month since registration)
        CASE 
            WHEN DATE_DIFF('month', c.registration_date, CURRENT_DATE) > 0 
            THEN COUNT(DISTINCT t.transaction_id) * 1.0 / DATE_DIFF('month', c.registration_date, CURRENT_DATE)
            ELSE 0 
        END as purchase_frequency_rate
        
    FROM nucentral_mysqldb.amato.customers_clean c
    LEFT JOIN nucentral_mysqldb.amato.transactions_clean t ON c.customer_id = t.customer_id
    LEFT JOIN nucentral_mysqldb.amato.transaction_items_clean ti ON t.transaction_id = ti.transaction_id
    WHERE t.order_status IN ('Delivered', 'Shipped')  -- Only successful orders
    GROUP BY 
        c.customer_id, c.email, c.first_name, c.last_name, c.registration_date
),

rfm_scores AS (
    SELECT 
        *,
        
        -- Recency Score (1-5, where 5 is most recent)
        CASE 
            WHEN recency_days <= 30 THEN 5
            WHEN recency_days <= 60 THEN 4
            WHEN recency_days <= 90 THEN 3
            WHEN recency_days <= 180 THEN 2
            ELSE 1
        END as recency_score,
        
        -- Frequency Score (1-5, where 5 is most frequent)
        CASE 
            WHEN frequency >= 10 THEN 5
            WHEN frequency >= 7 THEN 4
            WHEN frequency >= 4 THEN 3
            WHEN frequency >= 2 THEN 2
            ELSE 1
        END as frequency_score,
        
        -- Monetary Score (1-5, where 5 is highest value)
        CASE 
            WHEN monetary_value >= 5000 THEN 5
            WHEN monetary_value >= 3000 THEN 4
            WHEN monetary_value >= 1500 THEN 3
            WHEN monetary_value >= 500 THEN 2
            ELSE 1
        END as monetary_score
        
    FROM customer_metrics
),

rfm_segments AS (
    SELECT 
        *,
        
        -- Combined RFM Score (3-digit number)
        (recency_score * 100) + (frequency_score * 10) + monetary_score as rfm_score,
        
        -- RFM Segment based on combined score
        CASE 
            WHEN (recency_score * 100) + (frequency_score * 10) + monetary_score >= 500 THEN 'Champions'
            WHEN (recency_score * 100) + (frequency_score * 10) + monetary_score >= 400 THEN 'Loyal'
            WHEN (recency_score * 100) + (frequency_score * 10) + monetary_score >= 300 THEN 'At Risk'
            WHEN (recency_score * 100) + (frequency_score * 10) + monetary_score >= 200 THEN 'Can''t Lose'
            WHEN (recency_score * 100) + (frequency_score * 10) + monetary_score >= 100 THEN 'New'
            ELSE 'Promising'
        END as rfm_segment,
        
        -- Customer Value Tier
        CASE 
            WHEN monetary_value >= 5000 THEN 'Premium'
            WHEN monetary_value >= 2000 THEN 'High'
            WHEN monetary_value >= 1000 THEN 'Medium'
            WHEN monetary_value >= 500 THEN 'Low'
            ELSE 'Minimal'
        END as customer_value_tier,
        
        -- Engagement Level
        CASE 
            WHEN frequency >= 8 AND recency_days <= 60 THEN 'Highly Engaged'
            WHEN frequency >= 5 AND recency_days <= 90 THEN 'Engaged'
            WHEN frequency >= 3 AND recency_days <= 180 THEN 'Moderately Engaged'
            WHEN frequency >= 1 AND recency_days <= 365 THEN 'Low Engagement'
            ELSE 'Inactive'
        END as engagement_level
        
    FROM rfm_scores
)

SELECT * FROM rfm_segments;

-- 2. Create RFM Summary Statistics
CREATE TABLE IF NOT EXISTS nucentral_mysqldb.amato.rfm_summary_stats AS
SELECT 
    rfm_segment,
    customer_value_tier,
    engagement_level,
    COUNT(*) as customer_count,
    ROUND(AVG(monetary_value), 2) as avg_monetary_value,
    ROUND(AVG(frequency), 2) as avg_frequency,
    ROUND(AVG(recency_days), 1) as avg_recency_days,
    ROUND(AVG(customer_lifetime_value), 2) as avg_lifetime_value,
    ROUND(AVG(avg_order_value), 2) as avg_order_value,
    ROUND(AVG(purchase_frequency_rate), 2) as avg_purchase_frequency
FROM nucentral_mysqldb.amato.customer_rfm_data
GROUP BY rfm_segment, customer_value_tier, engagement_level
ORDER BY avg_monetary_value DESC;

-- 3. Display RFM Analysis Results
SELECT 
    rfm_segment,
    customer_count,
    ROUND(avg_monetary_value, 2) as avg_value,
    ROUND(avg_frequency, 2) as avg_orders,
    ROUND(avg_recency_days, 1) as avg_days_since_purchase
FROM nucentral_mysqldb.amato.rfm_summary_stats
ORDER BY avg_monetary_value DESC;
