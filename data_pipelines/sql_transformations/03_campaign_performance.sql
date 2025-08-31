-- AMATO Production - Campaign Performance Analysis SQL Script
-- This script analyzes campaign performance, ROI, and efficiency metrics
-- Run this script using Trino to evaluate marketing campaign effectiveness

-- Set the session schema
USE nucentral_postgresdb.amato;

-- 1. Campaign Performance Analysis with ROI Metrics
CREATE TABLE IF NOT EXISTS nucentral_postgresdb.amato.campaign_performance_data AS
WITH campaign_metrics AS (
    SELECT 
        c.campaign_id,
        c.campaign_name,
        c.campaign_type,
        c.channel,
        c.target_audience,
        c.start_date,
        c.end_date,
        c.budget,
        c.status,
        
        -- Performance aggregations
        SUM(cp.impressions) as total_impressions,
        SUM(cp.clicks) as total_clicks,
        SUM(cp.conversions) as total_conversions,
        SUM(cp.revenue) as total_revenue,
        
        -- Calculated metrics
        CASE 
            WHEN SUM(cp.impressions) > 0 THEN SUM(cp.clicks) * 1.0 / SUM(cp.impressions)
            ELSE 0 
        END as overall_ctr,
        
        CASE 
            WHEN SUM(cp.clicks) > 0 THEN SUM(cp.conversions) * 1.0 / SUM(cp.clicks)
            ELSE 0 
        END as overall_conversion_rate,
        
        CASE 
            WHEN c.budget > 0 THEN SUM(cp.revenue) * 1.0 / c.budget
            ELSE 0 
        END as overall_roas,
        
        -- Campaign duration
        DATE_DIFF('day', c.start_date, c.end_date) as campaign_duration,
        
        -- Daily averages
        CASE 
            WHEN DATE_DIFF('day', c.start_date, c.end_date) > 0 
            THEN SUM(cp.impressions) * 1.0 / DATE_DIFF('day', c.start_date, c.end_date)
            ELSE 0 
        END as avg_daily_impressions,
        
        CASE 
            WHEN DATE_DIFF('day', c.start_date, c.end_date) > 0 
            THEN SUM(cp.clicks) * 1.0 / DATE_DIFF('day', c.start_date, c.end_date)
            ELSE 0 
        END as avg_daily_clicks,
        
        CASE 
            WHEN DATE_DIFF('day', c.start_date, c.end_date) > 0 
            THEN SUM(cp.conversions) * 1.0 / DATE_DIFF('day', c.start_date, c.end_date)
            ELSE 0 
        END as avg_daily_conversions,
        
        CASE 
            WHEN DATE_DIFF('day', c.start_date, c.end_date) > 0 
            THEN SUM(cp.revenue) * 1.0 / DATE_DIFF('day', c.start_date, c.end_date)
            ELSE 0 
        END as avg_daily_revenue
        
    FROM nucentral_postgresdb.amato.campaigns_clean c
    LEFT JOIN nucentral_postgresdb.amato.campaign_performance_clean cp ON c.campaign_id = cp.campaign_id
    GROUP BY 
        c.campaign_id, c.campaign_name, c.campaign_type, c.channel, 
        c.target_audience, c.start_date, c.end_date, c.budget, c.status
),

campaign_segments AS (
    SELECT 
        *,
        
        -- Performance segmentation
        CASE 
            WHEN overall_roas >= 3.0 THEN 'High'
            WHEN overall_roas >= 2.0 THEN 'Medium'
            WHEN overall_roas >= 1.0 THEN 'Low'
            ELSE 'Poor'
        END as performance_segment,
        
        -- Efficiency score (0-100)
        CASE 
            WHEN overall_roas >= 5.0 THEN 100
            WHEN overall_roas >= 3.0 THEN 80
            WHEN overall_roas >= 2.0 THEN 60
            WHEN overall_roas >= 1.5 THEN 40
            WHEN overall_roas >= 1.0 THEN 20
            ELSE 0
        END as efficiency_score,
        
        -- Budget utilization
        CASE 
            WHEN budget > 0 THEN (total_revenue / budget) * 100
            ELSE 0 
        END as budget_utilization_percent,
        
        -- Rankings
        ROW_NUMBER() OVER (ORDER BY total_revenue DESC) as revenue_rank,
        ROW_NUMBER() OVER (ORDER BY total_conversions DESC) as conversion_rank,
        ROW_NUMBER() OVER (ORDER BY overall_roas DESC) as roas_rank
        
    FROM campaign_metrics
)

SELECT * FROM campaign_segments;

-- 4. Campaign Performance Summary by Channel
CREATE TABLE IF NOT EXISTS nucentral_postgresdb.amato.campaign_channel_summary AS
SELECT 
    channel,
    COUNT(*) as total_campaigns,
    SUM(budget) as total_budget,
    SUM(total_revenue) as total_revenue,
    AVG(overall_roas) as avg_roas,
    AVG(overall_ctr) as avg_ctr,
    AVG(overall_conversion_rate) as avg_conversion_rate,
    SUM(total_impressions) as total_impressions,
    SUM(total_clicks) as total_clicks,
    SUM(total_conversions) as total_conversions,
    
    -- Channel efficiency
    CASE 
        WHEN SUM(budget) > 0 THEN SUM(total_revenue) * 1.0 / SUM(budget)
        ELSE 0 
    END as channel_roas,
    
    -- Channel CTR
    CASE 
        WHEN SUM(total_impressions) > 0 THEN SUM(total_clicks) * 1.0 / SUM(total_impressions)
        ELSE 0 
    END as channel_ctr,
    
    -- Channel conversion rate
    CASE 
        WHEN SUM(total_clicks) > 0 THEN SUM(total_conversions) * 1.0 / SUM(total_clicks)
        ELSE 0 
    END as channel_conversion_rate
    
FROM nucentral_postgresdb.amato.campaign_performance_data
GROUP BY channel
ORDER BY channel_roas DESC;

-- 5. Campaign Performance Summary by Type
CREATE TABLE IF NOT EXISTS nucentral_postgresdb.amato.campaign_type_summary AS
SELECT 
    campaign_type,
    COUNT(*) as total_campaigns,
    SUM(budget) as total_budget,
    SUM(total_revenue) as total_revenue,
    AVG(overall_roas) as avg_roas,
    AVG(overall_ctr) as avg_ctr,
    AVG(overall_conversion_rate) as avg_conversion_rate,
    
    -- Type efficiency
    CASE 
        WHEN SUM(budget) > 0 THEN SUM(total_revenue) * 1.0 / SUM(budget)
        ELSE 0 
    END as type_roas
    
FROM nucentral_postgresdb.amato.campaign_performance_data
GROUP BY campaign_type
ORDER BY type_roas DESC;

-- 6. Display Campaign Performance Summary
SELECT 
    'Channel Performance' as summary_type,
    channel as category,
    total_campaigns,
    ROUND(total_budget, 2) as total_budget,
    ROUND(total_revenue, 2) as total_revenue,
    ROUND(channel_roas, 2) as roas,
    ROUND(channel_ctr * 100, 2) as ctr_percent,
    ROUND(channel_conversion_rate * 100, 2) as conversion_percent
FROM nucentral_postgresdb.amato.campaign_channel_summary
UNION ALL
SELECT 
    'Campaign Type Performance' as summary_type,
    campaign_type as category,
    total_campaigns,
    ROUND(total_budget, 2) as total_budget,
    ROUND(total_revenue, 2) as total_revenue,
    ROUND(type_roas, 2) as roas,
    ROUND(avg_ctr * 100, 2) as ctr_percent,
    ROUND(avg_conversion_rate * 100, 2) as conversion_percent
FROM nucentral_postgresdb.amato.campaign_type_summary
ORDER BY summary_type, roas DESC;
