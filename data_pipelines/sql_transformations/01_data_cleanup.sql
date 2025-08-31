-- AMATO Production - Data Cleanup SQL Script
-- This script performs data quality checks and cleaning across all databases
-- Run this script using Trino to clean data before further processing

-- 1. Clean Customer Data (MySQL)
CREATE TABLE IF NOT EXISTS nucentral_mysqldb.amato.customers_clean AS
SELECT 
    customer_id,
    email,
    first_name,
    last_name,
    date_of_birth,
    gender,
    city,
    state,
    country,
    postal_code,
    income_bracket,
    registration_date,
    last_login_date,
    is_active,
    created_at,
    updated_at
FROM nucentral_mysqldb.amato.customers
WHERE 
    customer_id IS NOT NULL 
    AND email IS NOT NULL 
    AND email LIKE '%@%'
    AND registration_date IS NOT NULL
    AND registration_date <= CURRENT_DATE;

-- 2. Clean Customer Demographics (MySQL)
CREATE TABLE IF NOT EXISTS nucentral_mysqldb.amato.customer_demographics_clean AS
SELECT 
    cd.customer_id,
    cd.age_group,
    cd.education_level,
    cd.occupation,
    cd.household_size,
    cd.marital_status,
    cd.interests,
    cd.lifestyle_segment,
    cd.created_at
FROM nucentral_mysqldb.amato.customer_demographics cd
INNER JOIN nucentral_mysqldb.amato.customers_clean c ON cd.customer_id = c.customer_id
WHERE 
    cd.customer_id IS NOT NULL
    AND cd.age_group IS NOT NULL;

-- 3. Clean Transaction Data (MySQL)
CREATE TABLE IF NOT EXISTS nucentral_mysqldb.amato.transactions_clean AS
SELECT 
    t.transaction_id,
    t.customer_id,
    t.order_date,
    t.total_amount,
    t.currency,
    t.payment_method,
    t.shipping_address,
    t.order_status,
    t.created_at
FROM nucentral_mysqldb.amato.transactions t
INNER JOIN nucentral_mysqldb.amato.customers_clean c ON t.customer_id = c.customer_id
WHERE 
    t.transaction_id IS NOT NULL
    AND t.customer_id IS NOT NULL
    AND t.order_date IS NOT NULL
    AND t.total_amount > 0
    AND t.order_date <= CURRENT_DATE;

-- 4. Clean Transaction Items (MySQL)
CREATE TABLE IF NOT EXISTS nucentral_mysqldb.amato.transaction_items_clean AS
SELECT 
    ti.item_id,
    ti.transaction_id,
    ti.product_id,
    ti.product_name,
    ti.category,
    ti.subcategory,
    ti.quantity,
    ti.unit_price,
    ti.total_price,
    ti.created_at
FROM nucentral_mysqldb.amato.transaction_items ti
INNER JOIN nucentral_mysqldb.amato.transactions_clean t ON ti.transaction_id = t.transaction_id
WHERE 
    ti.item_id IS NOT NULL
    AND ti.transaction_id IS NOT NULL
    AND ti.quantity > 0
    AND ti.unit_price > 0
    AND ti.total_price > 0;

-- 5. Clean Campaign Data (PostgreSQL)
CREATE TABLE IF NOT EXISTS nucentral_postgresdb.amato.campaigns_clean AS
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
    created_at,
    updated_at
FROM nucentral_postgresdb.amato.campaigns
WHERE 
    campaign_id IS NOT NULL
    AND campaign_name IS NOT NULL
    AND start_date IS NOT NULL
    AND end_date IS NOT NULL
    AND start_date <= end_date
    AND budget > 0;

-- 6. Clean Campaign Performance (PostgreSQL)
CREATE TABLE IF NOT EXISTS nucentral_postgresdb.amato.campaign_performance_clean AS
SELECT 
    cp.performance_id,
    cp.campaign_id,
    cp.date,
    cp.impressions,
    cp.clicks,
    cp.conversions,
    cp.revenue,
    cp.ctr,
    cp.conversion_rate,
    cp.roas,
    cp.created_at
FROM nucentral_postgresdb.amato.campaign_performance cp
INNER JOIN nucentral_postgresdb.amato.campaigns_clean c ON cp.campaign_id = c.campaign_id
WHERE 
    cp.performance_id IS NOT NULL
    AND cp.campaign_id IS NOT NULL
    AND cp.date IS NOT NULL
    AND cp.impressions >= 0
    AND cp.clicks >= 0
    AND cp.conversions >= 0
    AND cp.revenue >= 0;

-- 7. Clean A/B Test Data (PostgreSQL)
CREATE TABLE IF NOT EXISTS nucentral_postgresdb.amato.ab_tests_clean AS
SELECT 
    abt.test_id,
    abt.test_name,
    abt.campaign_id,
    abt.variant_a_description,
    abt.variant_b_description,
    abt.start_date,
    abt.end_date,
    abt.sample_size,
    abt.test_status,
    abt.created_at,
    abt.updated_at
FROM nucentral_postgresdb.amato.ab_tests abt
INNER JOIN nucentral_postgresdb.amato.campaigns_clean c ON abt.campaign_id = c.campaign_id
WHERE 
    abt.test_id IS NOT NULL
    AND abt.test_name IS NOT NULL
    AND abt.start_date IS NOT NULL
    AND abt.end_date IS NOT NULL
    AND abt.start_date <= abt.end_date
    AND abt.sample_size > 0;

-- 8. Clean A/B Test Results (PostgreSQL)
CREATE TABLE IF NOT EXISTS nucentral_postgresdb.amato.ab_test_results_clean AS
SELECT 
    abr.result_id,
    abr.test_id,
    abr.variant,
    abr.impressions,
    abr.clicks,
    abr.conversions,
    abr.revenue,
    abr.ctr,
    abr.conversion_rate,
    abr.created_at
FROM nucentral_postgresdb.amato.ab_test_results abr
INNER JOIN nucentral_postgresdb.amato.ab_tests_clean abt ON abr.test_id = abt.test_id
WHERE 
    abr.result_id IS NOT NULL
    AND abr.test_id IS NOT NULL
    AND abr.variant IN ('A', 'B')
    AND abr.impressions >= 0
    AND abr.clicks >= 0
    AND abr.conversions >= 0
    AND abr.revenue >= 0;

-- 9. Clean Clickstream Data (MongoDB)
-- Note: MongoDB data cleaning is done in the Python pipeline due to complex document structure
-- This is a placeholder for any SQL-based cleaning if needed

-- 10. Data Quality Summary
-- Create a summary table of cleaned data counts
CREATE TABLE IF NOT EXISTS nucentral_mysqldb.amato.data_quality_summary AS
SELECT 
    'customers' as table_name,
    COUNT(*) as clean_count,
    (SELECT COUNT(*) FROM nucentral_mysqldb.amato.customers) as original_count
FROM nucentral_mysqldb.amato.customers_clean
UNION ALL
SELECT 
    'transactions' as table_name,
    COUNT(*) as clean_count,
    (SELECT COUNT(*) FROM nucentral_mysqldb.amato.transactions) as original_count
FROM nucentral_mysqldb.amato.transactions_clean
UNION ALL
SELECT 
    'campaigns' as table_name,
    COUNT(*) as clean_count,
    (SELECT COUNT(*) FROM nucentral_postgresdb.amato.campaigns) as original_count
FROM nucentral_postgresdb.amato.campaigns_clean
UNION ALL
SELECT 
    'ab_tests' as table_name,
    COUNT(*) as clean_count,
    (SELECT COUNT(*) FROM nucentral_postgresdb.amato.ab_tests) as original_count
FROM nucentral_postgresdb.amato.ab_tests_clean;

-- Display data quality summary
SELECT * FROM nucentral_mysqldb.amato.data_quality_summary;
