#!/bin/bash

# Bokeh Effect Generator - Setup Script
# This script creates a virtual environment and installs all dependencies

set -e  # Exit on error

echo "🎨 Bokeh Effect Generator - Setup"
echo "=================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Display Python version
PYTHON_VERSION=$(python3 --version)
echo "✓ Found: $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
sudo apt update
sudo apt install -y python3.10-venv
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "✓ Virtual environment recreated"
    else
        echo "✓ Using existing virtual environment"
    fi
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip > /dev/null
echo "✓ pip upgraded"
echo ""

# Install requirements
echo "📥 Installing dependencies..."
echo "This may take several minutes..."
echo ""
pip install -r requirements.txt

echo ""
echo "=================================="
echo "✅ Setup complete!"
echo ""
echo "To run the application:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the application:"
echo "     python bokeh_app.py"
echo ""
echo "  3. Open your browser to:"
echo "     http://localhost:7860"
echo ""
echo "To deactivate the virtual environment when done:"
echo "  deactivate"
echo "=================================="
