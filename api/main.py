#!/usr/bin/env python3
"""
AMATO Production - FastAPI Application
Real-time inference API for customer analytics models
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import yaml
import logging
import os
import joblib
from datetime import datetime
from typing import List, Dict, Any, Optional
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AMATO Production API",
    description="Real-time inference API for customer analytics models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerData(BaseModel):
    """Customer data for inference"""
    customer_id: str = Field(..., description="Unique customer identifier")
    recency_days: float = Field(..., description="Days since last purchase")
    frequency: int = Field(..., description="Number of purchases")
    monetary_value: float = Field(..., description="Total amount spent")
    recency_score: int = Field(..., description="Recency score (1-5)")
    frequency_score: int = Field(..., description="Frequency score (1-5)")
    monetary_score: int = Field(..., description="Monetary score (1-5)")
    rfm_score: int = Field(..., description="Combined RFM score")
    customer_lifetime_value: float = Field(..., description="Customer lifetime value")
    total_sessions: int = Field(..., description="Total sessions")
    avg_session_duration: int = Field(..., description="Average session duration")
    total_page_views: int = Field(..., description="Total page views")
    conversion_events: int = Field(..., description="Conversion events")
    add_to_cart_events: int = Field(..., description="Add to cart events")
    search_queries: int = Field(..., description="Search queries")
    product_interactions: int = Field(..., description="Product interactions")
    unique_products_viewed: int = Field(..., description="Unique products viewed")
    avg_engagement_score: float = Field(..., description="Average engagement score")
    cart_abandonment_rate: float = Field(..., description="Cart abandonment rate")
    bounce_rate: float = Field(..., description="Bounce rate")
    return_visitor_rate: float = Field(..., description="Return visitor rate")
    purchase_velocity: float = Field(..., description="Purchase velocity")
    engagement_score_normalized: float = Field(..., description="Normalized engagement score")
    session_intensity: float = Field(..., description="Session intensity")
    conversion_efficiency: float = Field(..., description="Conversion efficiency")
    search_intensity: float = Field(..., description="Search intensity")
    product_exploration: float = Field(..., description="Product exploration")
    cart_behavior: float = Field(..., description="Cart behavior")
    rfm_composite: float = Field(..., description="RFM composite score")
    value_engagement_ratio: float = Field(..., description="Value engagement ratio")
    churn_risk: float = Field(..., description="Churn risk")
    upsell_potential: float = Field(..., description="Upsell potential")
    lifetime_value_potential: float = Field(..., description="Lifetime value potential")
    is_high_value: int = Field(..., description="High value customer flag")
    is_engaged: int = Field(..., description="Engaged customer flag")
    is_mobile_user: int = Field(..., description="Mobile user flag")
    is_return_visitor: int = Field(..., description="Return visitor flag")

class SegmentationRequest(BaseModel):
    """Request for customer segmentation"""
    customer_data: CustomerData
    model_type: str = Field(default="kmeans", description="Model type: kmeans or hdbscan")

class SegmentationResponse(BaseModel):
    """Response for customer segmentation"""
    customer_id: str
    segment: int
    segment_type: str
    confidence_score: float
    model_used: str
    inference_timestamp: str
    segment_characteristics: Dict[str, Any]

class ForecastingRequest(BaseModel):
    """Request for revenue forecasting"""
    customer_ids: List[str]
    forecast_horizon: int = Field(default=30, description="Forecast horizon in days")
    model_type: str = Field(default="prophet", description="Model type: prophet or xgboost")

class ForecastingResponse(BaseModel):
    """Response for revenue forecasting"""
    customer_id: str
    forecast_date: str
    predicted_revenue: float
    confidence_lower: float
    confidence_upper: float
    model_used: str
    inference_timestamp: str

class JourneySimulationRequest(BaseModel):
    """Request for journey simulation"""
    customer_id: str
    journey_type: str = Field(..., description="Journey type: purchase, browse, search")
    simulation_steps: int = Field(default=10, description="Number of simulation steps")

class JourneySimulationResponse(BaseModel):
    """Response for journey simulation"""
    customer_id: str
    journey_type: str
    predicted_path: List[str]
    conversion_probability: float
    expected_revenue: float
    model_used: str
    inference_timestamp: str

class CampaignOptimizationRequest(BaseModel):
    """Request for campaign optimization"""
    customer_segment: str
    campaign_type: str
    budget: float
    target_audience_size: int

class CampaignOptimizationResponse(BaseModel):
    """Response for campaign optimization"""
    customer_segment: str
    campaign_type: str
    recommended_budget: float
    expected_roas: float
    predicted_conversions: int
    optimal_channels: List[str]
    model_used: str
    inference_timestamp: str

class ModelManager:
    """Manages loaded ML models"""
    
    def __init__(self, config_path='config/database_config.yaml'):
        self.config = self.load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        self.load_models()
    
    def load_config(self, config_path):
        """Load configuration"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_models(self):
        """Load all trained models"""
        logger.info("🔄 Loading ML models...")
        
        model_path = self.config['ml']['model_storage_path']
        
        # Load customer segmentation models
        segmentation_path = os.path.join(model_path, 'customer_segmentation')
        if os.path.exists(segmentation_path):
            for model_file in os.listdir(segmentation_path):
                if model_file.endswith('_model.pkl'):
                    model_name = model_file.replace('_model.pkl', '')
                    model_path_full = os.path.join(segmentation_path, model_file)
                    scaler_path = os.path.join(segmentation_path, f'{model_name}_scaler.pkl')
                    metadata_path = os.path.join(segmentation_path, f'{model_name}_metadata.yaml')
                    
                    try:
                        self.models[f'segmentation_{model_name}'] = joblib.load(model_path_full)
                        if os.path.exists(scaler_path):
                            self.scalers[f'segmentation_{model_name}'] = joblib.load(scaler_path)
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                self.metadata[f'segmentation_{model_name}'] = yaml.safe_load(f)
                        
                        logger.info(f"✅ Loaded {model_name} segmentation model")
                    except Exception as e:
                        logger.error(f"❌ Failed to load {model_name} model: {e}")
        
        # Load other models (forecasting, journey simulation, campaign optimization)
        # These would be implemented similarly when those pipelines are created
        
        logger.info(f"✅ Loaded {len(self.models)} models")
    
    def get_model(self, model_type: str, model_name: str = None):
        """Get a specific model"""
        model_key = f"{model_type}_{model_name}" if model_name else model_type
        return self.models.get(model_key)
    
    def get_scaler(self, model_type: str, model_name: str = None):
        """Get a specific scaler"""
        scaler_key = f"{model_type}_{model_name}" if model_name else model_type
        return self.scalers.get(scaler_key)
    
    def get_metadata(self, model_type: str, model_name: str = None):
        """Get model metadata"""
        metadata_key = f"{model_type}_{model_name}" if model_name else model_type
        return self.metadata.get(metadata_key)

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("🚀 AMATO Production API starting up...")
    logger.info(f"📊 Loaded {len(model_manager.models)} models")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AMATO Production API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": len(model_manager.models)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(model_manager.models)
    }

@app.get("/models")
async def list_models():
    """List available models"""
    models_info = {}
    for model_key, model in model_manager.models.items():
        models_info[model_key] = {
            "type": type(model).__name__,
            "metadata": model_manager.metadata.get(model_key, {})
        }
    
    return {
        "models": models_info,
        "total_models": len(models_info)
    }

@app.post("/segment/customer", response_model=SegmentationResponse)
async def segment_customer(request: SegmentationRequest):
    """Segment a customer using trained models"""
    try:
        # Get model
        model = model_manager.get_model("segmentation", request.model_type)
        scaler = model_manager.get_scaler("segmentation", request.model_type)
        metadata = model_manager.get_metadata("segmentation", request.model_type)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {request.model_type} not found")
        
        # Prepare features
        customer_data = request.customer_data.dict()
        feature_columns = metadata.get('feature_columns', [])
        
        # Extract features in correct order
        features = []
        for col in feature_columns:
            if col in customer_data:
                features.append(customer_data[col])
            else:
                features.append(0.0)  # Default value for missing features
        
        # Scale features
        if scaler:
            features_scaled = scaler.transform([features])
        else:
            features_scaled = [features]
        
        # Make prediction
        segment = int(model.predict(features_scaled)[0])
        
        # Get segment characteristics
        segment_summary = metadata.get('segment_summary', [])
        segment_info = next((s for s in segment_summary if s['segment'] == segment), {})
        
        # Calculate confidence score (simplified)
        confidence_score = 0.8  # In production, this would be calculated based on model confidence
        
        return SegmentationResponse(
            customer_id=request.customer_data.customer_id,
            segment=segment,
            segment_type=segment_info.get('segment_type', 'Unknown'),
            confidence_score=confidence_score,
            model_used=request.model_type,
            inference_timestamp=datetime.now().isoformat(),
            segment_characteristics=segment_info
        )
        
    except Exception as e:
        logger.error(f"Error in customer segmentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment/batch", response_model=List[SegmentationResponse])
async def segment_customers_batch(customers: List[CustomerData], model_type: str = "kmeans"):
    """Segment multiple customers in batch"""
    try:
        results = []
        for customer_data in customers:
            request = SegmentationRequest(customer_data=customer_data, model_type=model_type)
            result = await segment_customer(request)
            results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch segmentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast/revenue", response_model=List[ForecastingResponse])
async def forecast_revenue(request: ForecastingRequest):
    """Forecast revenue for customers"""
    try:
        # This would be implemented when forecasting models are available
        # For now, return placeholder responses
        
        results = []
        for customer_id in request.customer_ids:
            # Placeholder forecasting logic
            base_revenue = 100.0
            forecast_revenue = base_revenue * (1 + np.random.normal(0, 0.1))
            
            results.append(ForecastingResponse(
                customer_id=customer_id,
                forecast_date=(datetime.now() + pd.Timedelta(days=request.forecast_horizon)).isoformat(),
                predicted_revenue=forecast_revenue,
                confidence_lower=forecast_revenue * 0.8,
                confidence_upper=forecast_revenue * 1.2,
                model_used=request.model_type,
                inference_timestamp=datetime.now().isoformat()
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Error in revenue forecasting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/journey/simulate", response_model=JourneySimulationResponse)
async def simulate_journey(request: JourneySimulationRequest):
    """Simulate customer journey"""
    try:
        # This would be implemented when journey simulation models are available
        # For now, return placeholder response
        
        journey_paths = {
            'purchase': ['Home', 'Products', 'Product Detail', 'Cart', 'Checkout'],
            'browse': ['Home', 'Categories', 'Product List', 'Product Detail'],
            'search': ['Home', 'Search', 'Search Results', 'Product Detail']
        }
        
        path = journey_paths.get(request.journey_type, ['Home', 'Products'])
        conversion_prob = 0.3 if request.journey_type == 'purchase' else 0.1
        expected_revenue = 150.0 if request.journey_type == 'purchase' else 0.0
        
        return JourneySimulationResponse(
            customer_id=request.customer_id,
            journey_type=request.journey_type,
            predicted_path=path[:request.simulation_steps],
            conversion_probability=conversion_prob,
            expected_revenue=expected_revenue,
            model_used="journey_simulation",
            inference_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in journey simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/campaign/optimize", response_model=CampaignOptimizationResponse)
async def optimize_campaign(request: CampaignOptimizationRequest):
    """Optimize campaign parameters"""
    try:
        # This would be implemented when campaign optimization models are available
        # For now, return placeholder response
        
        # Placeholder optimization logic
        recommended_budget = request.budget * 1.2
        expected_roas = 3.5
        predicted_conversions = int(request.target_audience_size * 0.05)
        optimal_channels = ['Email', 'Social Media', 'Search']
        
        return CampaignOptimizationResponse(
            customer_segment=request.customer_segment,
            campaign_type=request.campaign_type,
            recommended_budget=recommended_budget,
            expected_roas=expected_roas,
            predicted_conversions=predicted_conversions,
            optimal_channels=optimal_channels,
            model_used="campaign_optimization",
            inference_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in campaign optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get API metrics"""
    return {
        "total_models": len(model_manager.models),
        "api_version": "1.0.0",
        "uptime": "running",
        "endpoints": [
            "/segment/customer",
            "/segment/batch", 
            "/forecast/revenue",
            "/journey/simulate",
            "/campaign/optimize"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
