-- PostgreSQL Schema for AMATO Production
-- Campaign and A/B Test Data

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS amato;

-- Set search path to the schema
SET search_path TO amato;

-- Campaign tables
CREATE TABLE campaigns (
    campaign_id VARCHAR(50) PRIMARY KEY,
    campaign_name VARCHAR(255) NOT NULL,
    campaign_type VARCHAR(100),
    channel VARCHAR(50),
    target_audience VARCHAR(100),
    start_date DATE,
    end_date DATE,
    budget DECIMAL(12,2),
    status VARCHAR(20) DEFAULT 'Active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_campaigns_type ON campaigns(campaign_type);
CREATE INDEX idx_campaigns_channel ON campaigns(channel);
CREATE INDEX idx_campaigns_status ON campaigns(status);
CREATE INDEX idx_campaigns_dates ON campaigns(start_date, end_date);

CREATE TABLE campaign_performance (
    performance_id VARCHAR(50) PRIMARY KEY,
    campaign_id VARCHAR(50),
    date DATE,
    impressions INTEGER,
    clicks INTEGER,
    conversions INTEGER,
    revenue DECIMAL(10,2),
    ctr DECIMAL(5,4),
    conversion_rate DECIMAL(5,4),
    roas DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id) ON DELETE CASCADE
);

CREATE INDEX idx_campaign_performance_campaign ON campaign_performance(campaign_id);
CREATE INDEX idx_campaign_performance_date ON campaign_performance(date);
CREATE INDEX idx_campaign_performance_metrics ON campaign_performance(impressions, clicks, conversions);

-- A/B Test tables
CREATE TABLE ab_tests (
    test_id VARCHAR(50) PRIMARY KEY,
    test_name VARCHAR(255) NOT NULL,
    campaign_id VARCHAR(50),
    variant_a_description TEXT,
    variant_b_description TEXT,
    start_date DATE,
    end_date DATE,
    sample_size INTEGER,
    test_status VARCHAR(20) DEFAULT 'Running',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id) ON DELETE SET NULL
);

CREATE INDEX idx_ab_tests_campaign ON ab_tests(campaign_id);
CREATE INDEX idx_ab_tests_status ON ab_tests(test_status);
CREATE INDEX idx_ab_tests_dates ON ab_tests(start_date, end_date);

CREATE TABLE ab_test_results (
    result_id VARCHAR(50) PRIMARY KEY,
    test_id VARCHAR(50),
    variant VARCHAR(10),
    impressions INTEGER,
    clicks INTEGER,
    conversions INTEGER,
    revenue DECIMAL(10,2),
    ctr DECIMAL(5,4),
    conversion_rate DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (test_id) REFERENCES ab_tests(test_id) ON DELETE CASCADE
);

CREATE INDEX idx_ab_test_results_test ON ab_test_results(test_id);
CREATE INDEX idx_ab_test_results_variant ON ab_test_results(variant);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_campaigns_updated_at BEFORE UPDATE ON campaigns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ab_tests_updated_at BEFORE UPDATE ON ab_tests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
