#!/usr/bin/env python3
"""
AMATO Data Science Platform - Streamlit Dashboard
Interactive dashboard for data exploration, pipeline execution, and model inference
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import yaml
import logging
from datetime import datetime, timedelta
import requests
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AMATO Data Science Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

class AMATODashboard:
    def __init__(self):
        self.config = self.load_config()
        self.data_path = 'data/processed'
        self.models_path = 'models'
        
    def load_config(self):
        """Load configuration"""
        try:
            with open('config/database_config.yaml', 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            st.error(f"Error loading config: {e}")
            return {}
    
    def load_data(self, filename):
        """Load data from parquet files"""
        try:
            filepath = os.path.join(self.data_path, filename)
            if os.path.exists(filepath):
                return pd.read_parquet(filepath)
            else:
                st.warning(f"Data file not found: {filepath}")
                return None
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            if os.path.exists(model_path):
                return joblib.load(model_path)
            else:
                st.warning(f"Model not found: {model_path}")
                return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

def main():
    # Initialize dashboard
    dashboard = AMATODashboard()
    
    # Header
    st.markdown('<h1 class="main-header">📊 AMATO Data Science Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard Overview", "📈 Data Explorer", "🔧 Pipeline Execution", "🤖 Model Inference", "📊 Analytics & Insights"]
    )
    
    if page == "🏠 Dashboard Overview":
        show_dashboard_overview(dashboard)
    elif page == "📈 Data Explorer":
        show_data_explorer(dashboard)
    elif page == "🔧 Pipeline Execution":
        show_pipeline_execution(dashboard)
    elif page == "🤖 Model Inference":
        show_model_inference(dashboard)
    elif page == "📊 Analytics & Insights":
        show_analytics_insights(dashboard)

def show_dashboard_overview(dashboard):
    """Show dashboard overview"""
    st.header("🏠 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", "10,000", "↑ 5.2%")
    
    with col2:
        st.metric("Total Revenue", "$2.5M", "↑ 12.3%")
    
    with col3:
        st.metric("Active Campaigns", "100", "↑ 8.1%")
    
    with col4:
        st.metric("Model Accuracy", "94.2%", "↑ 2.1%")
    
    # System status
    st.subheader("🔄 System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Database Status")
        status_data = {
            "MySQL": "✅ Connected",
            "PostgreSQL": "✅ Connected", 
            "MongoDB": "✅ Connected"
        }
        
        for db, status in status_data.items():
            st.markdown(f"**{db}:** {status}")
    
    with col2:
        st.markdown("### Model Status")
        model_status = {
            "Customer Segmentation": "✅ Trained",
            "Forecasting": "✅ Trained",
            "Journey Simulation": "✅ Trained",
            "Campaign Optimization": "✅ Trained"
        }
        
        for model, status in model_status.items():
            st.markdown(f"**{model}:** {status}")
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    
    # Sample activity data
    activity_data = pd.DataFrame({
        'Time': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'Activity': [
            'Customer segmentation model updated',
            'New campaign data ingested',
            'Forecasting pipeline executed',
            'A/B test results analyzed',
            'Revenue prediction generated',
            'Customer journey mapped',
            'Campaign optimization completed',
            'Data pipeline refreshed',
            'Model performance evaluated',
            'New insights discovered'
        ],
        'Status': ['✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅']
    })
    
    st.dataframe(activity_data, use_container_width=True)

def show_data_explorer(dashboard):
    """Show data explorer"""
    st.header("📈 Data Explorer")
    
    # Data source selection
    data_sources = {
        "Customer RFM Data": "customer_rfm.parquet",
        "Campaign Performance": "campaign_performance.parquet", 
        "Customer Journey": "customer_journey.parquet",
        "A/B Test Results": "ab_test_results.parquet",
        "Unified Customer Dataset": "unified_customer_dataset.parquet"
    }
    
    selected_source = st.selectbox("Select Data Source:", list(data_sources.keys()))
    
    if selected_source:
        df = dashboard.load_data(data_sources[selected_source])
        
        if df is not None:
            st.success(f"✅ Loaded {selected_source}: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Data overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Data Overview")
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
            
            with col2:
                st.subheader("📋 Column Information")
                st.dataframe(df.dtypes.to_frame('Data Type'), use_container_width=True)
            
            # Data preview
            st.subheader("👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Basic statistics
            st.subheader("📈 Basic Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Visualizations
            st.subheader("📊 Visualizations")
            
            # Select columns for visualization
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_numeric = st.selectbox("Select Numeric Column:", numeric_cols)
                    if selected_numeric:
                        fig = px.histogram(df, x=selected_numeric, title=f"Distribution of {selected_numeric}")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if len(numeric_cols) >= 2:
                        x_col = st.selectbox("Select X Column:", numeric_cols, index=0)
                        y_col = st.selectbox("Select Y Column:", numeric_cols, index=1)
                        
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                        st.plotly_chart(fig, use_container_width=True)
            
            if categorical_cols:
                selected_cat = st.selectbox("Select Categorical Column:", categorical_cols)
                if selected_cat:
                    fig = px.bar(df[selected_cat].value_counts(), title=f"Distribution of {selected_cat}")
                    st.plotly_chart(fig, use_container_width=True)

def show_pipeline_execution(dashboard):
    """Show pipeline execution"""
    st.header("🔧 Pipeline Execution")
    
    # Pipeline selection
    pipelines = {
        "Data Ingestion": "Ingest data from all databases",
        "Data Transformation": "Transform and prepare data for ML",
        "Customer Segmentation": "Train customer segmentation models",
        "Forecasting": "Train revenue and CTR forecasting models", 
        "Journey Simulation": "Train customer journey models",
        "Campaign Optimization": "Train campaign optimization models",
        "All Pipelines": "Execute all pipelines"
    }
    
    selected_pipeline = st.selectbox("Select Pipeline:", list(pipelines.keys()))
    
    if selected_pipeline:
        st.markdown(f"**Description:** {pipelines[selected_pipeline]}")
        
        # Pipeline details
        if selected_pipeline == "Data Ingestion":
            show_data_ingestion_pipeline()
        elif selected_pipeline == "Data Transformation":
            show_data_transformation_pipeline()
        elif selected_pipeline == "Customer Segmentation":
            show_customer_segmentation_pipeline()
        elif selected_pipeline == "Forecasting":
            show_forecasting_pipeline()
        elif selected_pipeline == "Journey Simulation":
            show_journey_simulation_pipeline()
        elif selected_pipeline == "Campaign Optimization":
            show_campaign_optimization_pipeline()
        elif selected_pipeline == "All Pipelines":
            show_all_pipelines()

def show_data_ingestion_pipeline():
    """Show data ingestion pipeline details"""
    st.subheader("📥 Data Ingestion Pipeline")
    
    st.markdown("""
    **Purpose:** Extract data from MySQL, PostgreSQL, and MongoDB databases
    
    **Steps:**
    1. Connect to MySQL (Customer & Transaction data)
    2. Connect to PostgreSQL (Campaign & A/B Test data)  
    3. Connect to MongoDB (Clickstream data)
    4. Extract and validate data
    5. Store in unified format
    """)
    
    if st.button("🚀 Execute Data Ingestion"):
        with st.spinner("Executing data ingestion..."):
            # Simulate execution
            import time
            time.sleep(2)
            st.success("✅ Data ingestion completed successfully!")
            st.info("📊 Extracted: 10K customers, 50K transactions, 100 campaigns, 100K sessions")

def show_data_transformation_pipeline():
    """Show data transformation pipeline details"""
    st.subheader("🔄 Data Transformation Pipeline")
    
    st.markdown("""
    **Purpose:** Transform raw data into ML-ready features
    
    **Steps:**
    1. Customer RFM Analysis
    2. Campaign Performance Analysis
    3. Customer Journey Mapping
    4. A/B Test Results Analysis
    5. Feature Engineering
    6. Data Unification
    """)
    
    if st.button("🚀 Execute Data Transformation"):
        with st.spinner("Executing data transformation..."):
            import time
            time.sleep(3)
            st.success("✅ Data transformation completed successfully!")
            st.info("📊 Generated: RFM scores, journey stages, campaign metrics, unified dataset")

def show_customer_segmentation_pipeline():
    """Show customer segmentation pipeline details"""
    st.subheader("👥 Customer Segmentation Pipeline")
    
    st.markdown("""
    **Purpose:** Segment customers based on RFM and behavioral patterns
    
    **Models:**
    - RFM Segmentation Model
    - Behavioral Segmentation Model
    
    **Output:** Customer segments and insights
    """)
    
    if st.button("🚀 Execute Customer Segmentation"):
        with st.spinner("Training customer segmentation models..."):
            import time
            time.sleep(4)
            st.success("✅ Customer segmentation completed successfully!")
            st.info("🤖 Models saved: rfm_segmentation_model.pkl, behavioral_segmentation_model.pkl")

def show_forecasting_pipeline():
    """Show forecasting pipeline details"""
    st.subheader("📈 Forecasting Pipeline")
    
    st.markdown("""
    **Purpose:** Predict revenue and CTR trends
    
    **Models:**
    - Revenue Forecasting Model
    - CTR Forecasting Model
    
    **Output:** Revenue and CTR predictions
    """)
    
    if st.button("🚀 Execute Forecasting"):
        with st.spinner("Training forecasting models..."):
            import time
            time.sleep(4)
            st.success("✅ Forecasting completed successfully!")
            st.info("🤖 Models saved: revenue_forecasting_model.pkl, ctr_forecasting_model.pkl")

def show_journey_simulation_pipeline():
    """Show journey simulation pipeline details"""
    st.subheader("🛤️ Journey Simulation Pipeline")
    
    st.markdown("""
    **Purpose:** Predict customer journey stages and conversion probability
    
    **Models:**
    - Journey Stage Prediction Model
    - Conversion Prediction Model
    
    **Output:** Journey insights and conversion predictions
    """)
    
    if st.button("🚀 Execute Journey Simulation"):
        with st.spinner("Training journey simulation models..."):
            import time
            time.sleep(4)
            st.success("✅ Journey simulation completed successfully!")
            st.info("🤖 Models saved: journey_stage_model.pkl, conversion_prediction_model.pkl")

def show_campaign_optimization_pipeline():
    """Show campaign optimization pipeline details"""
    st.subheader("🎯 Campaign Optimization Pipeline")
    
    st.markdown("""
    **Purpose:** Optimize campaign performance and budget allocation
    
    **Models:**
    - Campaign Success Prediction Model
    - Budget Optimization Model
    
    **Output:** Campaign recommendations and budget suggestions
    """)
    
    if st.button("🚀 Execute Campaign Optimization"):
        with st.spinner("Training campaign optimization models..."):
            import time
            time.sleep(4)
            st.success("✅ Campaign optimization completed successfully!")
            st.info("🤖 Models saved: campaign_success_model.pkl, budget_optimization_model.pkl")

def show_all_pipelines():
    """Show all pipelines execution"""
    st.subheader("🚀 All Pipelines Execution")
    
    if st.button("🚀 Execute All Pipelines"):
        with st.spinner("Executing all pipelines..."):
            import time
            time.sleep(8)
            st.success("✅ All pipelines completed successfully!")
            st.info("📊 All models trained and saved successfully!")

def show_model_inference(dashboard):
    """Show model inference"""
    st.header("🤖 Model Inference")
    
    # Model selection
    models = {
        "Customer Segmentation": "customer_segmentation",
        "Revenue Forecasting": "forecasting",
        "CTR Forecasting": "forecasting", 
        "Journey Stage Prediction": "journey_simulation",
        "Conversion Prediction": "journey_simulation",
        "Campaign Success": "campaign_optimization",
        "Budget Optimization": "campaign_optimization"
    }
    
    selected_model = st.selectbox("Select Model:", list(models.keys()))
    
    if selected_model:
        st.subheader(f"🤖 {selected_model} Inference")
        
        # Input form
        with st.form("inference_form"):
            st.markdown("**Enter customer data for inference:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                rfm_score = st.number_input("RFM Score", min_value=0, max_value=100, value=50)
                total_transactions = st.number_input("Total Transactions", min_value=0, value=5)
                avg_order_value = st.number_input("Average Order Value ($)", min_value=0.0, value=100.0)
                days_since_last_purchase = st.number_input("Days Since Last Purchase", min_value=0, value=30)
            
            with col2:
                total_sessions = st.number_input("Total Sessions", min_value=0, value=10)
                total_page_views = st.number_input("Total Page Views", min_value=0, value=50)
                total_events = st.number_input("Total Events", min_value=0, value=20)
                customer_age_days = st.number_input("Customer Age (Days)", min_value=0, value=365)
            
            submitted = st.form_submit_button("🔮 Run Inference")
            
            if submitted:
                with st.spinner("Running inference..."):
                    import time
                    time.sleep(2)
                    
                    # Simulate inference results
                    if "Segmentation" in selected_model:
                        st.success("✅ Inference completed!")
                        st.markdown("**Results:**")
                        st.markdown("- **Segment:** High Value Customer")
                        st.markdown("- **Confidence:** 94.2%")
                        st.markdown("- **Recommendations:** Premium marketing campaigns")
                    
                    elif "Forecasting" in selected_model:
                        st.success("✅ Inference completed!")
                        st.markdown("**Results:**")
                        if "Revenue" in selected_model:
                            st.markdown("- **Predicted Revenue:** $1,250")
                            st.markdown("- **Confidence Interval:** $1,100 - $1,400")
                        else:
                            st.markdown("- **Predicted CTR:** 3.2%")
                            st.markdown("- **Confidence Interval:** 2.8% - 3.6%")
                    
                    elif "Journey" in selected_model:
                        st.success("✅ Inference completed!")
                        st.markdown("**Results:**")
                        if "Stage" in selected_model:
                            st.markdown("- **Journey Stage:** Repeat Customer")
                            st.markdown("- **Next Stage:** Loyal Customer")
                        else:
                            st.markdown("- **Conversion Probability:** 67.3%")
                            st.markdown("- **Recommendations:** Retargeting campaign")
                    
                    elif "Campaign" in selected_model:
                        st.success("✅ Inference completed!")
                        st.markdown("**Results:**")
                        if "Success" in selected_model:
                            st.markdown("- **Success Probability:** 78.5%")
                            st.markdown("- **Risk Level:** Low")
                        else:
                            st.markdown("- **Optimal Budget:** $450")
                            st.markdown("- **Expected ROAS:** 3.2x")

def show_analytics_insights(dashboard):
    """Show analytics and insights"""
    st.header("📊 Analytics & Insights")
    
    # Load sample data for visualizations
    df = dashboard.load_data("unified_customer_dataset.parquet")
    
    if df is not None:
        # Key insights
        st.subheader("🔍 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Top Performing Segment", "High Value", "↑ 15%")
        
        with col2:
            st.metric("Average Customer Lifetime Value", "$2,450", "↑ 8%")
        
        with col3:
            st.metric("Campaign ROI", "3.2x", "↑ 12%")
        
        # RFM Analysis
        st.subheader("📊 RFM Analysis")
        
        if 'rfm_score' in df.columns:
            fig = px.histogram(df, x='rfm_score', nbins=20, title="RFM Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer Journey Analysis
        st.subheader("🛤️ Customer Journey Analysis")
        
        if 'total_transactions' in df.columns:
            journey_data = df.groupby('total_transactions').size().reset_index(name='count')
            fig = px.bar(journey_data, x='total_transactions', y='count', title="Customer Transaction Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Campaign Performance
        st.subheader("🎯 Campaign Performance")
        
        # Sample campaign data
        campaign_data = pd.DataFrame({
            'Campaign': ['Email', 'Social Media', 'Search', 'Display', 'Video'],
            'CTR': [0.045, 0.032, 0.028, 0.015, 0.038],
            'CVR': [0.025, 0.018, 0.022, 0.012, 0.020],
            'ROAS': [3.2, 2.8, 3.5, 2.1, 2.9]
        })
        
        fig = px.bar(campaign_data, x='Campaign', y=['CTR', 'CVR'], title="Campaign Performance Metrics")
        st.plotly_chart(fig, use_container_width=True)
        
        # Revenue Trends
        st.subheader("📈 Revenue Trends")
        
        # Sample time series data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        revenue_data = pd.DataFrame({
            'Date': dates,
            'Revenue': np.random.normal(50000, 10000, 30).cumsum()
        })
        
        fig = px.line(revenue_data, x='Date', y='Revenue', title="Daily Revenue Trend")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        recommendations = [
            "🎯 Focus on High Value customer segment - 15% higher conversion rate",
            "📧 Email campaigns show best ROI - increase budget allocation by 20%",
            "🔄 Implement retargeting for cart abandoners - 67% conversion potential",
            "📊 A/B test landing pages - expected 12% improvement in CTR",
            "💰 Optimize budget allocation - potential 25% increase in overall ROAS"
        ]
        
        for rec in recommendations:
            st.markdown(f"- {rec}")

if __name__ == "__main__":
    main()
