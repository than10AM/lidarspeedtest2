#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Setting up environment..."

# 1. Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "Virtual environment 'venv' already exists. Skipping creation."
fi

# 2. Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# 3. Install requirements
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the benchmark
echo "Running the benchmark..."
python warp_raycast_test.py

echo "Done!"
