# 🗄️ Database Data Insertion Commands

## 📋 Prerequisites
Make sure you have the following installed:
- Python 3.8+
- Required Python packages (install with `pip install -r requirements.txt`)
- Database connections are accessible

## 🗄️ Database Setup Commands

### **1. Setup All Database Schemas (Recommended)**
```bash
# Setup all database schemas using Python script
python setup_database_schema.py
```

### **2. Clean All Database Tables (Before Data Generation)**
```bash
# Clean all existing data before generating new data
python clean_database.py
```

### **3. Manual Database Setup (Alternative)**
```bash
# MySQL Schema Setup
mysql -h 5.tcp.ngrok.io -P 27931 -u root -p'Password123!' < ddl/mysql_schema.sql

# PostgreSQL Schema Setup
psql -h 3.tcp.ngrok.io -p 27200 -U nuscaleadmin -d nuscale -f ddl/postgresql_schema.sql

# MongoDB Collections Setup
mongo mongodb://nuscaleadmin:Password123!@3.tcp.ngrok.io:21923/nuscale ddl/mongodb_schema.js
```

## 🚀 Individual Database Data Insertion Commands

### 1. **MySQL Data Insertion**

```bash
# Insert customer and transaction data into MySQL
python data_generation/mysql_data_generator.py
```

**What this does:**
- Connects to MySQL at `5.tcp.ngrok.io:27931`
- Database: `amato`
- User: `root`
- Generates:
  - 10,000 customers with demographics
  - 50,000 transactions
  - 150,000 transaction items
  - Customer segments and mappings

**Expected Output:**
```
INFO:__main__:Connected to MySQL database: amato
INFO:__main__:Generating 10000 customers...
INFO:__main__:✅ Generated 10000 customers
INFO:__main__:Generating 50000 transactions...
INFO:__main__:✅ Generated 50000 transactions
INFO:__main__:Generating 150000 transaction items...
INFO:__main__:✅ Generated 150000 transaction items
INFO:__main__:Generating customer segments...
INFO:__main__:✅ Generated 5 customer segments
INFO:__main__:Generating customer segment mappings...
INFO:__main__:✅ Generated 10000 customer segment mappings
INFO:__main__:MySQL data generation completed successfully!
```

### 2. **PostgreSQL Data Insertion**

```bash
# Insert campaign and A/B test data into PostgreSQL
python data_generation/postgresql_data_generator.py
```

**What this does:**
- Connects to PostgreSQL at `3.tcp.ngrok.io:27200`
- Database: `nuscale`
- Schema: `amato`
- User: `nuscaleadmin`
- Generates:
  - 100 campaigns
  - 1,000 campaign performance records
  - 50 A/B tests
  - 100 A/B test results

**Expected Output:**
```
INFO:__main__:Connected to PostgreSQL database: nuscale with schema: amato
INFO:__main__:Generating 100 campaigns...
INFO:__main__:✅ Generated 100 campaigns
INFO:__main__:Generating 1000 campaign performance records...
INFO:__main__:✅ Generated 1000 campaign performance records
INFO:__main__:Generating 50 A/B tests...
INFO:__main__:✅ Generated 50 A/B tests
INFO:__main__:Generating 100 A/B test results...
INFO:__main__:✅ Generated 100 A/B test results
INFO:__main__:PostgreSQL data generation completed successfully!
```

### 3. **MongoDB Data Insertion**

```bash
# Insert clickstream data into MongoDB
python data_generation/mongodb_data_generator.py
```

**What this does:**
- Connects to MongoDB at `3.tcp.ngrok.io:21923`
- Database: `amato`
- Auth Database: `admin`
- User: `nuscaleadmin`
- Generates:
  - 100,000 sessions
  - 800,000 page views
  - 600,000 events
  - 400,000 product interactions
  - 300,000 search queries

**Expected Output:**
```
INFO:__main__:Connected to MongoDB database: amato
INFO:__main__:Generating 100000 sessions...
INFO:__main__:✅ Generated 100000 sessions
INFO:__main__:Generating 800000 page views...
INFO:__main__:✅ Generated 800000 page views
INFO:__main__:Generating 600000 events...
INFO:__main__:✅ Generated 600000 events
INFO:__main__:Generating 400000 product interactions...
INFO:__main__:✅ Generated 400000 product interactions
INFO:__main__:Generating 300000 search queries...
INFO:__main__:✅ Generated 300000 search queries
INFO:__main__:MongoDB data generation completed successfully!
```

### 4. **All Databases at Once (Orchestrator)**

```bash
# Insert data into all databases simultaneously
python data_generation/generate_all_data.py
```

**What this does:**
- Runs all three data generators in sequence
- Provides a summary of all generated data
- Handles any errors gracefully

**Expected Output:**
```
INFO:__main__:Starting data generation for all databases...
INFO:__main__:=== MySQL Data Generation ===
[MySQL generation output]
INFO:__main__:=== PostgreSQL Data Generation ===
[PostgreSQL generation output]
INFO:__main__:=== MongoDB Data Generation ===
[MongoDB generation output]
INFO:__main__:All data generation completed successfully!
INFO:__main__:Summary:
- MySQL: 10,000 customers, 50,000 transactions, 150,000 items
- PostgreSQL: 100 campaigns, 1,000 performance records, 50 A/B tests
- MongoDB: 100,000 sessions, 800,000 page views, 600,000 events
```

## 🔧 Troubleshooting Commands

### Test Database Connections

```bash
# Test MySQL connection
python -c "
import mysql.connector
conn = mysql.connector.connect(
    host='5.tcp.ngrok.io',
    port=27931,
    database='amato',
    user='root',
    password='Password123!'
)
print('✅ MySQL connection successful')
conn.close()
"

# Test PostgreSQL connection
python -c "
import psycopg2
conn = psycopg2.connect(
    host='3.tcp.ngrok.io',
    port=27200,
    database='nuscale',
    user='nuscaleadmin',
    password='Password123!'
)
cursor = conn.cursor()
cursor.execute('SET search_path TO amato')
print('✅ PostgreSQL connection successful')
conn.close()
"

# Test MongoDB connection
python -c "
import pymongo
client = pymongo.MongoClient('mongodb://nuscaleadmin:Password123!@3.tcp.ngrok.io:21923/nuscale')
db = client['amato']
print('✅ MongoDB connection successful')
client.close()
"
```

### Check Generated Data

```bash
# Check MySQL data
python -c "
import mysql.connector
conn = mysql.connector.connect(
    host='5.tcp.ngrok.io',
    port=27931,
    database='amato',
    user='root',
    password='Password123!'
)
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM customers')
print(f'Customers: {cursor.fetchone()[0]}')
cursor.execute('SELECT COUNT(*) FROM transactions')
print(f'Transactions: {cursor.fetchone()[0]}')
conn.close()
"

# Check PostgreSQL data
python -c "
import psycopg2
conn = psycopg2.connect(
    host='3.tcp.ngrok.io',
    port=27200,
    database='nuscale',
    user='nuscaleadmin',
    password='Password123!'
)
cursor = conn.cursor()
cursor.execute('SET search_path TO amato')
cursor.execute('SELECT COUNT(*) FROM campaigns')
print(f'Campaigns: {cursor.fetchone()[0]}')
cursor.execute('SELECT COUNT(*) FROM ab_tests')
print(f'A/B Tests: {cursor.fetchone()[0]}')
conn.close()
"

# Check MongoDB data
python -c "
import pymongo
client = pymongo.MongoClient('mongodb://nuscaleadmin:Password123!@3.tcp.ngrok.io:21923/nuscale')
db = client['amato']
print(f'Sessions: {db.sessions.count_documents({})}')
print(f'Page Views: {db.page_views.count_documents({})}')
print(f'Events: {db.events.count_documents({})}')
client.close()
"
```

## 📊 Expected Data Volumes

After running all commands, you should have:

### MySQL Database (`amato`)
- **customers**: 10,000 records
- **customer_demographics**: 10,000 records
- **transactions**: 50,000 records
- **transaction_items**: 150,000 records
- **customer_segments**: 5 records
- **customer_segment_mapping**: 10,000 records

### PostgreSQL Database (`nuscale.amato`)
- **campaigns**: 100 records
- **campaign_performance**: 1,000 records
- **ab_tests**: 50 records
- **ab_test_results**: 100 records

### MongoDB Database (`amato`)
- **sessions**: 100,000 documents
- **page_views**: 800,000 documents
- **events**: 600,000 documents
- **product_interactions**: 400,000 documents
- **search_queries**: 300,000 documents

## ⚠️ Important Notes

1. **Order of Execution**: You can run the commands in any order, but it's recommended to run them individually first to test connections.

2. **Error Handling**: Each script includes error handling and will provide clear error messages if connections fail.

3. **Data Volume**: The scripts generate substantial amounts of data. Ensure your databases have sufficient storage.

4. **Network Latency**: Since you're using ngrok tunnels, expect some latency during data insertion.

5. **Time Estimates**: 
   - MySQL: ~2-3 minutes
   - PostgreSQL: ~1-2 minutes
   - MongoDB: ~5-10 minutes
   - All together: ~10-15 minutes

## 🤖 ML Pipeline Training Commands

### **1. Train All ML Pipelines**

```bash
# Train all ML pipelines at once
python train_all_ml_pipelines.py
```

**What this does:**
- Trains Customer Segmentation models
- Trains Forecasting models (Revenue & CTR)
- Trains Journey Simulation models
- Trains Campaign Optimization models
- Saves all models as .pkl files

### **2. Train Individual ML Pipelines**

```bash
# Customer Segmentation
python ml_pipelines/customer_segmentation/train_segmentation_models.py

# Forecasting
python ml_pipelines/forecasting/train_forecasting_models.py

# Journey Simulation
python ml_pipelines/journey_simulation/train_journey_models.py

# Campaign Optimization
python ml_pipelines/campaign_optimization/train_campaign_models.py
```

## 📊 Streamlit Dashboard Commands

### **1. Launch Streamlit Dashboard**

```bash
# Launch the interactive dashboard
streamlit run streamlit_dashboard.py
```

**What this provides:**
- 🏠 Dashboard Overview with key metrics
- 📈 Data Explorer for all datasets
- 🔧 Pipeline Execution interface
- 🤖 Model Inference interface
- 📊 Analytics & Insights with visualizations

**Access:** http://localhost:8501

## 🎯 Next Steps

After successfully inserting data:

1. **Run SQL Transformations**: Execute Trino SQL scripts in `data_pipelines/sql_transformations/`
2. **Create Unified Dataset**: Run `python data_pipelines/unified_dataset/create_unified_dataset.py`
3. **Train ML Models**: Run `python train_all_ml_pipelines.py`
4. **Launch Dashboard**: Run `streamlit run streamlit_dashboard.py`
5. **Start API Server**: Run `uvicorn api.main:app --reload`
