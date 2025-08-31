# 🚀 AMATO Production Data Science Platform

A comprehensive data science platform for customer analytics, featuring data pipelines, ML pipelines, and model development across multiple databases.

## 📋 Overview

AMATO Production is a multi-database data science platform that processes customer data from MySQL, PostgreSQL, and MongoDB to generate insights through machine learning models.

### 🎯 Key Features

- **Multi-Database Architecture**: MySQL (Customer/Transaction), PostgreSQL (Campaign/A/B Tests), MongoDB (Clickstream)
- **Data Pipelines**: SQL transformations using Trino, unified dataset creation
- **ML Pipelines**: Customer Segmentation, Forecasting, Journey Simulation, Campaign Optimization
- **Real-time Inference**: FastAPI endpoints for model predictions
- **Interactive Dashboard**: Streamlit application for data exploration and pipeline execution

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     MySQL       │    │   PostgreSQL    │    │     MongoDB     │
│                 │    │                 │    │                 │
│ • Customers     │    │ • Campaigns     │    │ • Sessions      │
│ • Transactions  │    │ • A/B Tests     │    │ • Page Views    │
│ • Demographics  │    │ • Performance   │    │ • Events        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │     Trino       │
                    │                 │
                    │ • SQL Queries   │
                    │ • Data Joins    │
                    │ • Transformations│
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Data Pipelines │
                    │                 │
                    │ • RFM Analysis  │
                    │ • Journey Data  │
                    │ • Performance   │
                    │ • A/B Results   │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │  ML Pipelines   │
                    │                 │
                    │ • Segmentation  │
                    │ • Forecasting   │
                    │ • Simulation    │
                    │ • Optimization  │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   FastAPI       │
                    │                 │
                    │ • Real-time     │
                    │ • Batch         │
                    │ • Inference     │
                    └─────────────────┘
```

## 📁 Project Structure

```
amato_production/
├── ddl/                          # Database schema files
│   ├── mysql_schema.sql         # MySQL DDL
│   ├── postgresql_schema.sql    # PostgreSQL DDL
│   └── mongodb_schema.js        # MongoDB schema
├── data_generation/             # Data generation scripts
│   ├── mysql_data_generator.py
│   ├── postgresql_data_generator.py
│   ├── mongodb_data_generator.py
│   └── generate_all_data.py
├── data_pipelines/              # Data transformation pipelines
│   ├── sql_transformations/     # Trino SQL scripts
│   └── unified_dataset/         # Python scripts for final dataset
├── ml_pipelines/                # Machine learning pipelines
│   ├── customer_segmentation/
│   ├── forecasting/
│   ├── journey_simulation/
│   └── campaign_optimization/
├── api/                         # FastAPI application
├── config/                      # Configuration files
│   └── database_config.yaml
├── docs/                        # Documentation
├── streamlit_app/               # Streamlit dashboard
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- MySQL 8.0+
- PostgreSQL 13+ (or install via Homebrew: `brew install postgresql`)
- MongoDB 5.0+
- Trino (for unified SQL queries)

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd amato_production

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Note: If you encounter PostgreSQL installation issues on macOS, run:
# brew install postgresql
# Then try installing requirements again
```

### 3. Database Setup

#### MySQL Setup
```bash
# Create database and user
mysql -u root -p
CREATE DATABASE amato_production;
CREATE USER 'amato_user'@'localhost' IDENTIFIED BY 'amato_password';
GRANT ALL PRIVILEGES ON amato_production.* TO 'amato_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;

# Run DDL
mysql -u amato_user -p amato_production < ddl/mysql_schema.sql
```

#### PostgreSQL Setup
```bash
# Create database and user
sudo -u postgres psql
CREATE DATABASE amato_production;
CREATE USER amato_user WITH PASSWORD 'amato_password';
GRANT ALL PRIVILEGES ON DATABASE amato_production TO amato_user;
\q

# Run DDL
psql -U amato_user -d amato_production -f ddl/postgresql_schema.sql
```

#### MongoDB Setup
```bash
# Start MongoDB
mongod

# Run schema script
mongo < ddl/mongodb_schema.js
```

### 4. Data Generation

```bash
# Generate all data
python data_generation/generate_all_data.py
```

### 5. Run Data Pipelines

```bash
# Run SQL transformations (Trino)
# (Trino setup required)

# Run unified dataset creation
python data_pipelines/unified_dataset/create_unified_dataset.py
```

### 6. Run ML Pipelines

```bash
# Customer Segmentation
python ml_pipelines/customer_segmentation/train_segmentation_models.py

# Forecasting
python ml_pipelines/forecasting/train_forecasting_models.py

# Journey Simulation
python ml_pipelines/journey_simulation/train_journey_models.py

# Campaign Optimization
python ml_pipelines/campaign_optimization/train_optimization_models.py
```

### 7. Start API Server

```bash
# Start FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 8. Start Streamlit Dashboard

```bash
# Start Streamlit app
streamlit run streamlit_app/app.py
```

## 📊 Data Flow

### 1. Data Sources
- **MySQL**: Customer profiles, transactions, demographics
- **PostgreSQL**: Marketing campaigns, A/B test results
- **MongoDB**: Clickstream data (sessions, page views, events)

### 2. Data Processing
- **Trino SQL**: Cross-database queries and transformations
- **Python**: Data cleaning, feature engineering, aggregation

### 3. ML Models
- **Customer Segmentation**: K-means, HDBSCAN clustering
- **Forecasting**: Prophet, XGBoost time series
- **Journey Simulation**: LightGBM, Random Forest
- **Campaign Optimization**: XGBoost, Random Forest

### 4. Outputs
- **Parquet Files**: Processed datasets
- **Pickle Files**: Trained models
- **API Endpoints**: Real-time inference
- **Dashboard**: Interactive exploration

## 🔧 Configuration

Edit `config/database_config.yaml` to configure:
- Database connections
- Data generation parameters
- ML model settings
- API configuration

## 📈 Use Cases

### 1. Customer Segmentation
- **Input**: Customer RFM scores, behavioral patterns
- **Output**: Customer segments with characteristics
- **Business Value**: Targeted marketing, personalized experiences

### 2. Revenue Forecasting
- **Input**: Historical transaction data, campaign performance
- **Output**: Revenue predictions with confidence intervals
- **Business Value**: Budget planning, growth projections

### 3. Customer Journey Analysis
- **Input**: Clickstream data, conversion events
- **Output**: Journey patterns, conversion optimization
- **Business Value**: UX improvement, conversion rate optimization

### 4. Campaign Optimization
- **Input**: A/B test results, campaign performance
- **Output**: Optimal campaign strategies
- **Business Value**: Marketing efficiency, ROI improvement

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run specific pipeline tests
python -m pytest tests/test_data_pipelines.py
python -m pytest tests/test_ml_pipelines.py
```

## 📝 Logging

Logs are stored in `logs/` directory:
- `data_generation.log`: Data generation process
- `pipeline_execution.log`: Pipeline execution logs
- `api_access.log`: API access logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation in `docs/`
- Review the logs for error details

## 🔄 Version History

- **v1.0.0**: Initial production release
  - Multi-database architecture
  - Complete data pipelines
  - ML model training and inference
  - Interactive dashboard
