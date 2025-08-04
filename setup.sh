#!/bin/bash

echo "🚀 vLLM Research Project Setup"
echo "================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda first:"
    echo "   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "   chmod +x Miniconda3-latest-Linux-x86_64.sh"
    echo "   ./Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# Check if NVIDIA GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  NVIDIA GPU not detected. vLLM requires CUDA-capable GPU."
    echo "   Please ensure you have NVIDIA drivers and CUDA installed."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create conda environment
echo "📦 Creating conda environment..."
conda create -n vllm_research python=3.10 -y

# Activate environment
echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm_research

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup completed!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate vllm_research"
echo "2. Run the example: ./run_example.sh"
echo "3. Check the README.md for detailed usage instructions"
echo ""
echo "Happy researching! 🎓" 