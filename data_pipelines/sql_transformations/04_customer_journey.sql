-- AMATO Production - Customer Journey Analysis SQL Script
-- This script analyzes customer journey patterns from clickstream data
-- Run this script using Trino to understand customer behavior and engagement

-- Set the session schema
USE nucentral_mysqldb.amato;

-- 1. Customer Journey Analysis from Clickstream Data
CREATE TABLE IF NOT EXISTS nucentral_mysqldb.amato.customer_journey_data AS
WITH session_analysis AS (
    SELECT 
        s.customer_id,
        COUNT(DISTINCT s.session_id) as total_sessions,
        AVG(s.duration_seconds) as avg_session_duration,
        SUM(s.duration_seconds) as total_session_time,
        COUNT(DISTINCT s.device_type) as device_types_used,
        COUNT(DISTINCT s.browser) as browsers_used,
        
        -- Device preference
        CASE 
            WHEN COUNT(CASE WHEN s.device_type = 'Mobile' THEN 1 END) > COUNT(CASE WHEN s.device_type = 'Desktop' THEN 1 END) THEN 'Mobile'
            WHEN COUNT(CASE WHEN s.device_type = 'Desktop' THEN 1 END) > COUNT(CASE WHEN s.device_type = 'Mobile' THEN 1 END) THEN 'Desktop'
            ELSE 'Mixed'
        END as device_preference,
        
        -- Browser preference
        CASE 
            WHEN COUNT(CASE WHEN s.browser = 'Chrome' THEN 1 END) > COUNT(CASE WHEN s.browser != 'Chrome' THEN 1 END) THEN 'Chrome'
            WHEN COUNT(CASE WHEN s.browser = 'Safari' THEN 1 END) > COUNT(CASE WHEN s.browser != 'Safari' THEN 1 END) THEN 'Safari'
            WHEN COUNT(CASE WHEN s.browser = 'Firefox' THEN 1 END) > COUNT(CASE WHEN s.browser != 'Firefox' THEN 1 END) THEN 'Firefox'
            ELSE 'Other'
        END as browser_preference
        
    FROM nucentral_mongodb.amato.sessions s
    WHERE s.customer_id IS NOT NULL
    GROUP BY s.customer_id
),

page_view_analysis AS (
    SELECT 
        pv.customer_id,
        COUNT(DISTINCT pv.view_id) as total_page_views,
        COUNT(DISTINCT pv.page_url) as unique_pages_viewed,
        AVG(pv.time_on_page) as avg_time_on_page,
        SUM(pv.time_on_page) as total_time_on_pages,
        
        -- Engagement metrics
        CASE 
            WHEN AVG(pv.scroll_depth) >= 80 THEN 'High'
            WHEN AVG(pv.scroll_depth) >= 50 THEN 'Medium'
            ELSE 'Low'
        END as scroll_engagement
        
    FROM nucentral_mongodb.amato.page_views pv
    WHERE pv.session_id IS NOT NULL
    GROUP BY pv.customer_id
),

event_analysis AS (
    SELECT 
        e.customer_id,
        COUNT(DISTINCT e.event_id) as total_events,
        COUNT(CASE WHEN e.event_type = 'click' THEN 1 END) as click_events,
        COUNT(CASE WHEN e.event_type = 'scroll' THEN 1 END) as scroll_events,
        COUNT(CASE WHEN e.event_type = 'form_submit' THEN 1 END) as form_submit_events,
        COUNT(CASE WHEN e.event_type = 'add_to_cart' THEN 1 END) as add_to_cart_events,
        COUNT(CASE WHEN e.event_type = 'purchase' THEN 1 END) as purchase_events,
        
        -- Event category
        CASE 
            WHEN COUNT(CASE WHEN e.event_type IN ('purchase', 'add_to_cart') THEN 1 END) > 0 THEN 'Converter'
            WHEN COUNT(CASE WHEN e.event_type = 'form_submit' THEN 1 END) > 0 THEN 'Engaged'
            WHEN COUNT(CASE WHEN e.event_type = 'click' THEN 1 END) > 5 THEN 'Active'
            ELSE 'Passive'
        END as event_category
        
    FROM nucentral_mongodb.amato.events e
    WHERE e.session_id IS NOT NULL
    GROUP BY e.customer_id
),

product_interaction_analysis AS (
    SELECT 
        pi.customer_id,
        COUNT(DISTINCT pi.interaction_id) as total_interactions,
        COUNT(DISTINCT pi.product_id) as unique_products_viewed,
        COUNT(CASE WHEN pi.interaction_type = 'view' THEN 1 END) as product_views,
        COUNT(CASE WHEN pi.interaction_type = 'add_to_cart' THEN 1 END) as cart_adds,
        COUNT(CASE WHEN pi.interaction_type = 'wishlist' THEN 1 END) as wishlist_adds,
        
        -- Interaction intensity
        CASE 
            WHEN COUNT(DISTINCT pi.product_id) >= 10 THEN 'High'
            WHEN COUNT(DISTINCT pi.product_id) >= 5 THEN 'Medium'
            ELSE 'Low'
        END as interaction_intensity
        
    FROM nucentral_mongodb.amato.product_interactions pi
    WHERE pi.session_id IS NOT NULL
    GROUP BY pi.customer_id
),

search_analysis AS (
    SELECT 
        sq.customer_id,
        COUNT(DISTINCT sq.query_id) as total_search_queries,
        COUNT(DISTINCT sq.search_term) as unique_search_terms,
        COUNT(CASE WHEN sq.clicked_result_position IS NOT NULL THEN 1 END) as clicked_search_results,
        
        -- Search effectiveness
        CASE 
            WHEN COUNT(CASE WHEN sq.clicked_result_position IS NOT NULL THEN 1 END) > COUNT(DISTINCT sq.query_id) * 0.5 THEN 'Effective'
            WHEN COUNT(CASE WHEN sq.clicked_result_position IS NOT NULL THEN 1 END) > 0 THEN 'Moderate'
            ELSE 'Ineffective'
        END as search_effectiveness
        
    FROM nucentral_mongodb.amato.search_queries sq
    WHERE sq.session_id IS NOT NULL
    GROUP BY sq.customer_id
),

journey_patterns AS (
    SELECT 
        c.customer_id,
        c.email,
        c.first_name,
        c.last_name,
        
        -- Session metrics
        COALESCE(sa.total_sessions, 0) as total_sessions,
        COALESCE(sa.avg_session_duration, 0) as avg_session_duration,
        COALESCE(sa.device_preference, 'Unknown') as device_preference,
        COALESCE(sa.browser_preference, 'Unknown') as browser_preference,
        
        -- Page view metrics
        COALESCE(pva.total_page_views, 0) as total_page_views,
        COALESCE(pva.unique_pages_viewed, 0) as unique_pages_viewed,
        COALESCE(pva.avg_time_on_page, 0) as avg_time_on_page,
        COALESCE(pva.scroll_engagement, 'Unknown') as scroll_engagement,
        
        -- Event metrics
        COALESCE(ea.total_events, 0) as total_events,
        COALESCE(ea.click_events, 0) as click_events,
        COALESCE(ea.add_to_cart_events, 0) as add_to_cart_events,
        COALESCE(ea.purchase_events, 0) as purchase_events,
        COALESCE(ea.event_category, 'Unknown') as event_category,
        
        -- Product interaction metrics
        COALESCE(pia.total_interactions, 0) as total_interactions,
        COALESCE(pia.unique_products_viewed, 0) as unique_products_viewed,
        COALESCE(pia.product_views, 0) as product_views,
        COALESCE(pia.cart_adds, 0) as cart_adds,
        COALESCE(pia.interaction_intensity, 'Unknown') as interaction_intensity,
        
        -- Search metrics
        COALESCE(sqa.total_search_queries, 0) as total_search_queries,
        COALESCE(sqa.unique_search_terms, 0) as unique_search_terms,
        COALESCE(sqa.clicked_search_results, 0) as clicked_search_results,
        COALESCE(sqa.search_effectiveness, 'Unknown') as search_effectiveness,
        
        -- Journey classification
        CASE 
            WHEN COALESCE(ea.purchase_events, 0) > 0 THEN 'Converter'
            WHEN COALESCE(ea.add_to_cart_events, 0) > 0 THEN 'Engaged'
            WHEN COALESCE(sa.total_sessions, 0) > 5 THEN 'Explorer'
            WHEN COALESCE(sa.total_sessions, 0) > 1 THEN 'Browser'
            ELSE 'Visitor'
        END as journey_type,
        
        -- Conversion status
        CASE 
            WHEN COALESCE(ea.purchase_events, 0) > 0 THEN 'Converted'
            WHEN COALESCE(ea.add_to_cart_events, 0) > 0 THEN 'Abandoned'
            ELSE 'Browsing'
        END as conversion_status,
        
        -- Journey complexity
        CASE 
            WHEN COALESCE(sa.total_sessions, 0) > 10 AND COALESCE(pva.unique_pages_viewed, 0) > 20 THEN 'Complex'
            WHEN COALESCE(sa.total_sessions, 0) > 5 AND COALESCE(pva.unique_pages_viewed, 0) > 10 THEN 'Medium'
            ELSE 'Simple'
        END as journey_complexity,
        
        -- Engagement score (0-100)
        CASE 
            WHEN COALESCE(ea.purchase_events, 0) > 0 THEN 100
            WHEN COALESCE(ea.add_to_cart_events, 0) > 0 THEN 80
            WHEN COALESCE(ea.click_events, 0) > 10 THEN 60
            WHEN COALESCE(sa.total_sessions, 0) > 5 THEN 40
            WHEN COALESCE(sa.total_sessions, 0) > 1 THEN 20
            ELSE 0
        END as engagement_score,
        
        -- Conversion probability (0-1)
        CASE 
            WHEN COALESCE(ea.purchase_events, 0) > 0 THEN 1.0
            WHEN COALESCE(ea.add_to_cart_events, 0) > 0 THEN 0.7
            WHEN COALESCE(ea.click_events, 0) > 10 THEN 0.5
            WHEN COALESCE(sa.total_sessions, 0) > 5 THEN 0.3
            WHEN COALESCE(sa.total_sessions, 0) > 1 THEN 0.1
            ELSE 0.0
        END as conversion_probability
        
    FROM nucentral_mysqldb.amato.customers_clean c
    LEFT JOIN session_analysis sa ON c.customer_id = sa.customer_id
    LEFT JOIN page_view_analysis pva ON c.customer_id = pva.customer_id
    LEFT JOIN event_analysis ea ON c.customer_id = ea.customer_id
    LEFT JOIN product_interaction_analysis pia ON c.customer_id = pia.customer_id
    LEFT JOIN search_analysis sqa ON c.customer_id = sqa.customer_id
)

SELECT * FROM journey_patterns;

-- 5. Journey Pattern Summary
CREATE TABLE IF NOT EXISTS nucentral_mysqldb.amato.journey_pattern_summary AS
SELECT 
    journey_type,
    conversion_status,
    journey_complexity,
    device_preference,
    browser_preference,
    COUNT(*) as session_count,
    ROUND(AVG(engagement_score), 2) as avg_engagement_score,
    ROUND(AVG(conversion_probability), 3) as avg_conversion_probability,
    ROUND(AVG(total_sessions), 2) as avg_sessions,
    ROUND(AVG(total_page_views), 2) as avg_page_views,
    ROUND(AVG(avg_session_duration), 2) as avg_session_duration,
    ROUND(AVG(unique_products_viewed), 2) as avg_products_viewed,
    ROUND(AVG(total_search_queries), 2) as avg_search_queries,
    COUNT(CASE WHEN purchase_events > 0 THEN 1 END) * 1.0 / COUNT(*) as conversion_rate,
    COUNT(CASE WHEN add_to_cart_events > 0 THEN 1 END) * 1.0 / COUNT(*) as cart_add_rate
    
FROM nucentral_mysqldb.amato.customer_journey_data
GROUP BY 
    journey_type, conversion_status, journey_complexity, 
    device_preference, browser_preference
ORDER BY session_count DESC;

-- 6. Display Journey Analysis Results
SELECT 
    journey_type,
    conversion_status,
    session_count,
    ROUND(avg_engagement_score, 1) as avg_engagement,
    ROUND(conversion_rate * 100, 2) as conversion_rate_percent,
    ROUND(cart_add_rate * 100, 2) as cart_add_rate_percent
FROM nucentral_mysqldb.amato.journey_pattern_summary
ORDER BY session_count DESC;
