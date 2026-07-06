#!/bin/bash
echo "Installing requirements..."
pip install -r requirements.txt

echo "Fixing transformers version..."
pip install "transformers>=4.57.0,<5.0.0"

echo "Installing hf_transfer..."
pip install hf_transfer

echo "Setup complete!"
