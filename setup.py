#!/usr/bin/env python3
"""
AMATO Production Setup Script
Automates the setup process for the AMATO production environment
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AMATOSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            sys.exit(1)
        logger.info(f"✅ Python version: {sys.version}")
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            "logs",
            "data/processed",
            "models",
            "features",
            "tests"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
    
    def create_virtual_environment(self):
        """Create Python virtual environment"""
        if self.venv_path.exists():
            logger.info("✅ Virtual environment already exists")
            return
        
        try:
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)
            logger.info("✅ Virtual environment created")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            sys.exit(1)
    
    def install_dependencies(self):
        """Install Python dependencies"""
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            logger.error("requirements.txt not found")
            sys.exit(1)
        
        # Determine pip path
        if os.name == 'nt':  # Windows
            pip_path = self.venv_path / "Scripts" / "pip"
        else:  # Unix/Linux/macOS
            pip_path = self.venv_path / "bin" / "pip"
        
        try:
            subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], check=True)
            logger.info("✅ Dependencies installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            sys.exit(1)
    
    def create_config_template(self):
        """Create configuration template if it doesn't exist"""
        config_file = self.project_root / "config" / "database_config.yaml"
        if config_file.exists():
            logger.info("✅ Configuration file already exists")
            return
        
        config_template = """# AMATO Production Database Configuration
# Update these values with your actual database credentials

trino:
  host: "localhost"
  port: 8080
  catalog: "hive"
  schema: "default"
  username: "trino"
  password: ""

databases:
  mysql:
    host: "localhost"
    port: 3306
    database: "amato_production"
    username: "amato_user"
    password: "amato_password"
    charset: "utf8mb4"
    
  postgresql:
    host: "localhost"
    port: 5432
    database: "amato_production"
    username: "amato_user"
    password: "amato_password"
    schema: "public"
    
  mongodb:
    host: "localhost"
    port: 27017
    database: "amato_production"
    username: "amato_user"
    password: "amato_password"
    auth_source: "admin"

# Data generation settings
data_generation:
  customers_count: 1000  # Reduced for testing
  transactions_count: 5000
  campaigns_count: 50
  ab_tests_count: 25
  sessions_count: 10000
  page_views_count: 80000
  events_count: 60000
  product_interactions_count: 40000
  search_queries_count: 30000

# Other settings remain the same as in the original config
"""
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            f.write(config_template)
        
        logger.info("✅ Configuration template created")
        logger.info("⚠️  Please update config/database_config.yaml with your database credentials")
    
    def create_gitignore(self):
        """Create .gitignore file"""
        gitignore_file = self.project_root / ".gitignore"
        if gitignore_file.exists():
            logger.info("✅ .gitignore already exists")
            return
        
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/
*.log

# Data
data/
*.parquet
*.csv
*.json

# Models
models/
*.pkl
*.joblib

# Environment variables
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Testing
.coverage
.pytest_cache/
htmlcov/
"""
        
        with open(gitignore_file, 'w') as f:
            f.write(gitignore_content)
        
        logger.info("✅ .gitignore created")
    
    def run_setup(self):
        """Run complete setup process"""
        logger.info("🚀 Starting AMATO Production Setup...")
        
        # Check Python version
        self.check_python_version()
        
        # Create directories
        self.create_directories()
        
        # Create virtual environment
        self.create_virtual_environment()
        
        # Install dependencies
        self.install_dependencies()
        
        # Create configuration template
        self.create_config_template()
        
        # Create .gitignore
        self.create_gitignore()
        
        logger.info("=" * 60)
        logger.info("🎉 AMATO Production Setup Completed!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("📋 Next Steps:")
        logger.info("1. Update config/database_config.yaml with your database credentials")
        logger.info("2. Set up your databases (MySQL, PostgreSQL, MongoDB)")
        logger.info("3. Run the DDL scripts in the ddl/ directory")
        logger.info("4. Activate the virtual environment:")
        if os.name == 'nt':  # Windows
            logger.info("   venv\\Scripts\\activate")
        else:  # Unix/Linux/macOS
            logger.info("   source venv/bin/activate")
        logger.info("5. Generate data: python data_generation/generate_all_data.py")
        logger.info("")
        logger.info("📚 For detailed instructions, see README.md")

def main():
    """Main function"""
    try:
        setup = AMATOSetup()
        setup.run_setup()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
