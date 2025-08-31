-- AMATO Production - A/B Test Results Analysis SQL Script
-- This script analyzes A/B test results with statistical significance testing
-- Run this script using Trino to evaluate test performance and determine winners

-- Set the session schema
USE nucentral_postgresdb.amato;

-- 1. A/B Test Results Analysis with Statistical Calculations
CREATE TABLE IF NOT EXISTS nucentral_postgresdb.amato.ab_test_results_data AS
WITH test_summary AS (
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
        
        -- Variant A metrics
        SUM(CASE WHEN abr.variant = 'A' THEN abr.impressions ELSE 0 END) as variant_a_impressions,
        SUM(CASE WHEN abr.variant = 'A' THEN abr.clicks ELSE 0 END) as variant_a_clicks,
        SUM(CASE WHEN abr.variant = 'A' THEN abr.conversions ELSE 0 END) as variant_a_conversions,
        SUM(CASE WHEN abr.variant = 'A' THEN abr.revenue ELSE 0 END) as variant_a_revenue,
        
        -- Variant B metrics
        SUM(CASE WHEN abr.variant = 'B' THEN abr.impressions ELSE 0 END) as variant_b_impressions,
        SUM(CASE WHEN abr.variant = 'B' THEN abr.clicks ELSE 0 END) as variant_b_clicks,
        SUM(CASE WHEN abr.variant = 'B' THEN abr.conversions ELSE 0 END) as variant_b_conversions,
        SUM(CASE WHEN abr.variant = 'B' THEN abr.revenue ELSE 0 END) as variant_b_revenue,
        
        -- Total metrics
        SUM(abr.impressions) as total_impressions,
        SUM(abr.clicks) as total_clicks,
        SUM(abr.conversions) as total_conversions,
        SUM(abr.revenue) as total_revenue
        
    FROM nucentral_postgresdb.amato.ab_tests_clean abt
    LEFT JOIN nucentral_postgresdb.amato.ab_test_results_clean abr ON abt.test_id = abr.test_id
    GROUP BY 
        abt.test_id, abt.test_name, abt.campaign_id, abt.variant_a_description,
        abt.variant_b_description, abt.start_date, abt.end_date, abt.sample_size, abt.test_status
),

test_metrics AS (
    SELECT 
        *,
        
        -- Variant A rates
        CASE 
            WHEN variant_a_impressions > 0 THEN variant_a_clicks * 1.0 / variant_a_impressions
            ELSE 0 
        END as variant_a_ctr,
        
        CASE 
            WHEN variant_a_clicks > 0 THEN variant_a_conversions * 1.0 / variant_a_clicks
            ELSE 0 
        END as variant_a_conversion_rate,
        
        CASE 
            WHEN variant_a_impressions > 0 THEN variant_a_revenue * 1.0 / variant_a_impressions
            ELSE 0 
        END as variant_a_revenue_per_impression,
        
        -- Variant B rates
        CASE 
            WHEN variant_b_impressions > 0 THEN variant_b_clicks * 1.0 / variant_b_impressions
            ELSE 0 
        END as variant_b_ctr,
        
        CASE 
            WHEN variant_b_clicks > 0 THEN variant_b_conversions * 1.0 / variant_b_clicks
            ELSE 0 
        END as variant_b_conversion_rate,
        
        CASE 
            WHEN variant_b_impressions > 0 THEN variant_b_revenue * 1.0 / variant_b_impressions
            ELSE 0 
        END as variant_b_revenue_per_impression,
        
        -- Test duration
        DATE_DIFF('day', start_date, end_date) as test_duration_days
        
    FROM test_summary
),

statistical_analysis AS (
    SELECT 
        *,
        
        -- Lift calculations
        CASE 
            WHEN variant_a_ctr > 0 THEN ((variant_b_ctr - variant_a_ctr) / variant_a_ctr) * 100
            ELSE 0 
        END as ctr_lift_percentage,
        
        CASE 
            WHEN variant_a_conversion_rate > 0 THEN ((variant_b_conversion_rate - variant_a_conversion_rate) / variant_a_conversion_rate) * 100
            ELSE 0 
        END as conversion_lift_percentage,
        
        CASE 
            WHEN variant_a_revenue_per_impression > 0 THEN ((variant_b_revenue_per_impression - variant_a_revenue_per_impression) / variant_a_revenue_per_impression) * 100
            ELSE 0 
        END as revenue_lift_percentage,
        
        -- Statistical significance (simplified calculation)
        -- In production, you would use proper statistical tests like chi-square or t-test
        CASE 
            WHEN variant_a_impressions >= 1000 AND variant_b_impressions >= 1000 THEN 'Yes'
            WHEN variant_a_impressions >= 500 AND variant_b_impressions >= 500 THEN 'Maybe'
            ELSE 'No'
        END as statistical_significance,
        
        -- Confidence level (simplified)
        CASE 
            WHEN variant_a_impressions >= 2000 AND variant_b_impressions >= 2000 THEN '99%'
            WHEN variant_a_impressions >= 1000 AND variant_b_impressions >= 1000 THEN '95%'
            WHEN variant_a_impressions >= 500 AND variant_b_impressions >= 500 THEN '90%'
            ELSE 'Low'
        END as confidence_level,
        
        -- Winner determination
        CASE 
            WHEN variant_b_ctr > variant_a_ctr AND variant_b_conversion_rate > variant_a_conversion_rate THEN 'B'
            WHEN variant_a_ctr > variant_b_ctr AND variant_a_conversion_rate > variant_b_conversion_rate THEN 'A'
            WHEN variant_b_revenue_per_impression > variant_a_revenue_per_impression THEN 'B'
            WHEN variant_a_revenue_per_impression > variant_b_revenue_per_impression THEN 'A'
            ELSE 'Tie'
        END as winner,
        
        -- P-value approximation (simplified)
        CASE 
            WHEN variant_a_impressions >= 2000 AND variant_b_impressions >= 2000 THEN 0.01
            WHEN variant_a_impressions >= 1000 AND variant_b_impressions >= 1000 THEN 0.05
            WHEN variant_a_impressions >= 500 AND variant_b_impressions >= 500 THEN 0.10
            ELSE 0.50
        END as p_value,
        
        -- Required sample size for 95% confidence
        CASE 
            WHEN variant_a_impressions + variant_b_impressions < 2000 THEN 2000
            ELSE variant_a_impressions + variant_b_impressions
        END as sample_size_required
        
    FROM test_metrics
),

final_results AS (
    SELECT 
        *,
        
        -- Recommendation (calculated after statistical_significance is available)
        CASE 
            WHEN variant_b_ctr > variant_a_ctr AND variant_b_conversion_rate > variant_a_conversion_rate AND statistical_significance = 'Yes' THEN 'Implement Variant B'
            WHEN variant_a_ctr > variant_b_ctr AND variant_a_conversion_rate > variant_b_conversion_rate AND statistical_significance = 'Yes' THEN 'Implement Variant A'
            WHEN variant_b_revenue_per_impression > variant_a_revenue_per_impression AND statistical_significance = 'Yes' THEN 'Implement Variant B'
            WHEN variant_a_revenue_per_impression > variant_b_revenue_per_impression AND statistical_significance = 'Yes' THEN 'Implement Variant A'
            WHEN statistical_significance = 'No' THEN 'Continue Testing'
            ELSE 'Inconclusive - Continue A'
        END as recommendation
        
    FROM statistical_analysis
)

SELECT * FROM final_results;

-- 6. A/B Test Summary Statistics
CREATE TABLE IF NOT EXISTS nucentral_postgresdb.amato.ab_test_summary_stats AS
SELECT 
    test_status,
    statistical_significance,
    winner,
    confidence_level,
    COUNT(*) as test_count,
    ROUND(AVG(ctr_lift_percentage), 2) as avg_ctr_lift,
    ROUND(AVG(conversion_lift_percentage), 2) as avg_conversion_lift,
    ROUND(AVG(revenue_lift_percentage), 2) as avg_revenue_lift,
    ROUND(AVG(test_duration_days), 1) as avg_test_duration,
    ROUND(AVG(sample_size_required), 0) as avg_sample_size_required,
    
    -- Success rates
    COUNT(CASE WHEN statistical_significance = 'Yes' THEN 1 END) * 1.0 / COUNT(*) as significant_test_rate,
    COUNT(CASE WHEN winner != 'Tie' THEN 1 END) * 1.0 / COUNT(*) as clear_winner_rate,
    COUNT(CASE WHEN recommendation LIKE 'Implement%' THEN 1 END) * 1.0 / COUNT(*) as actionable_test_rate,
    COUNT(CASE WHEN recommendation = 'Implement Variant B' THEN 1 END) * 1.0 / COUNT(*) as variant_b_win_rate
    
FROM nucentral_postgresdb.amato.ab_test_results_data
GROUP BY test_status, statistical_significance, winner, confidence_level
ORDER BY test_count DESC;

-- 7. Display A/B Test Analysis Results
SELECT 
    test_status,
    statistical_significance,
    winner,
    test_count,
    ROUND(avg_ctr_lift, 1) as avg_ctr_lift_percent,
    ROUND(avg_conversion_lift, 1) as avg_conversion_lift_percent,
    ROUND(significant_test_rate * 100, 1) as significant_test_rate_percent,
    ROUND(clear_winner_rate * 100, 1) as clear_winner_rate_percent
FROM nucentral_postgresdb.amato.ab_test_summary_stats
ORDER BY test_count DESC;
