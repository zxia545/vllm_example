# vLLM example

This repository contains a research framework for conducting psychological surveys using Large Language Models (LLMs) with persona-based prompting. The project enables researchers to generate survey responses from AI models that are instructed to roleplay as specific human personas, allowing for controlled studies of personality traits and survey responses.

## 📁 Project Structure

```
vllm_example/
├── dataset/
│   ├── original_dataset/          # Raw survey data from external sources
│   │   ├── NPI/                   # Narcissistic Personality Inventory
│   │   └── OSRI44_dev_data/       # Other survey data
│   ├── processed_dataset/         # Formatted survey questions and prompts
│   │   ├── NPI.jsonl             # NPI survey with formatted prompts
│   │   └── OSRI44_dev_data.jsonl # Other surveys with formatted prompts
│   └── persona_dataset/           # Human persona profiles
│       └── example_persona.jsonl  # Example persona data for testing
├── utils/                         # Utility functions for vLLM operations
│   └── utils.py                   # Core utilities for server management and API calls
├── generate_with_persona/         # Main generation scripts
│   └── generate_with_persona.py   # Script to run surveys with personas
└── README.md                      # This file
```

## 🎯 Overview

### Dataset Components

1. **Original Dataset** (`dataset/original_dataset/`)

   - Contains raw survey data downloaded from external sources
   - `codebook.txt`: Survey descriptions and question mappings
   - `data.csv`: Human response data for validation and comparison
2. **Processed Dataset** (`dataset/processed_dataset/`)

   - Contains formatted survey questions ready for LLM processing
   - Each JSONL file contains survey items with multiple prompt formats:
     - `prompt_with_persona_system`: System prompt for persona-based responses
     - `prompt_with_persona_user`: User prompt for persona-based responses
     - `prompt_direct_system`: System prompt for direct AI responses
     - `prompt_direct_user`: User prompt for direct AI responses
   - Includes original question metadata for analysis
3. **Persona Dataset** (`dataset/persona_dataset/`)

   - Contains human persona profiles for roleplaying experiments
   - Each persona includes:
     - `persona`: Text description of the person's characteristics
     - Personality trait scores (EXT, EST, AGR, CSN, OPN)
     - `unique_id`: Unique identifier for tracking responses
     - `source`: Origin of the persona data

### Core Components

1. **Utils** (`utils/utils.py`)

   - vLLM server management functions
   - API communication utilities
   - File I/O operations for JSONL files
   - GPU allocation and process management
2. **Generation Scripts** (`generate_with_persona/`)

   - Main script for running surveys with persona-based prompting
   - Supports batch processing of multiple personas
   - Handles response collection and output formatting

## 🚀 Quick Start Guide

### Prerequisites

- Linux system (tested on Ubuntu 20.04+)
- NVIDIA GPU with CUDA support
- Python 3.8+
- At least 16GB RAM (32GB+ recommended for larger models)

### 1. Install Miniconda

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Make executable and run
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

# Follow the installation prompts
# Restart your terminal or run:
source ~/.bashrc
```

### 2. Create and Activate Environment

```bash
# Create new conda environment
conda create -n vllm_research python=3.10

# Activate the environment
conda activate vllm_research
```

### 3. Install vLLM

```bash
# Install vLLM with CUDA support
pip install vllm

# Install additional dependencies
pip install openai requests tqdm
```

### 4. Test Installation

Create a simple test script to verify your setup:

```bash
# Create test script
cat > test_vllm.py << 'EOF'
#!/usr/bin/env python3

import sys
import subprocess
import time
import requests
from openai import OpenAI

def test_vllm_setup():
    print("Testing vLLM setup...")
  
    # Start vLLM server with Qwen2.5-0.5B model
    print("Starting vLLM server...")
    process = subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--port", "8000",
        "--host", "0.0.0.0"
    ])
  
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(30)
  
    try:
        # Test API connection
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
      
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            messages=[
                {"role": "user", "content": "Hello! Please respond with 'vLLM is working!'"}
            ],
            max_tokens=50,
            temperature=0.1
        )
      
        print(f"✅ vLLM is working! Response: {response.choices[0].message.content}")
      
    except Exception as e:
        print(f"❌ Error: {e}")
  
    finally:
        # Clean up
        process.terminate()
        process.wait()
        print("Server stopped.")

if __name__ == "__main__":
    test_vllm_setup()
EOF

# Make executable and run
chmod +x test_vllm.py
python test_vllm.py
```

## 📋 Running Experiments

### Basic Usage

1. **Start vLLM Server (You don't need to run this the genreate_with_persona.py run that in subprocess)**

```bash
# Start server with Qwen2.5-0.5B model
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --port 8000 \
    --host 0.0.0.0
```

2. **Run Survey Generation**

```bash
# Run NPI survey with persona dataset
python generate_with_persona/generate_with_persona.py \
    --input_file dataset/processed_dataset/NPI.jsonl \
    --output_file results/npi_persona_responses.jsonl \
    --persona_file dataset/persona_dataset/example_persona.jsonl \
    --api_base http://localhost:8000 \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --max_tokens 10 \
    --temperature 0.1 \
    --threads 4
```

### Example Shell Script

Create a complete example script:

```bash
# Create run_example.sh
cat > run_example.sh << 'EOF'
#!/bin/bash

# Configuration
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
PORT=8000
API_BASE="http://localhost:${PORT}"
INPUT_FILE="dataset/processed_dataset/NPI.jsonl"
OUTPUT_FILE="results/npi_persona_responses.jsonl"
PERSONA_FILE="dataset/persona_dataset/example_persona.jsonl"

# Create results directory
mkdir -p results

echo "🚀 Starting vLLM Research Example"
echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"

# Start vLLM server in background
echo "📡 Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --port ${PORT} \
    --host 0.0.0.0 > server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# Wait for server to start
echo "⏳ Waiting for server to start..."
sleep 30

# Check if server is running
if curl -s ${API_BASE}/health > /dev/null; then
    echo "✅ Server is running"
else
    echo "❌ Server failed to start"
    kill ${SERVER_PID} 2>/dev/null
    exit 1
fi

# Run the generation script
echo "🔬 Running survey generation..."
python generate_with_persona/generate_with_persona.py \
    --input_file ${INPUT_FILE} \
    --output_file ${OUTPUT_FILE} \
    --persona_file ${PERSONA_FILE} \
    --api_base ${API_BASE} \
    --model_name ${MODEL_NAME} \
    --max_tokens 10 \
    --temperature 0.1 \
    --threads 4

# Check results
if [ -f "${OUTPUT_FILE}" ]; then
    echo "✅ Results saved to ${OUTPUT_FILE}"
    echo "📊 Number of responses: $(wc -l < ${OUTPUT_FILE})"
else
    echo "❌ No results file generated"
fi

# Stop server
echo "🛑 Stopping server..."
kill ${SERVER_PID} 2>/dev/null
wait ${SERVER_PID} 2>/dev/null

echo "🎉 Example completed!"
EOF

# Make executable
chmod +x run_example.sh

# Run the example
./run_example.sh
```

## 🔧 Utils Functions Reference

### Core Functions in `utils/utils.py`

#### File Operations

- `read_jsonl(file_path)`: Read JSONL file with UTF-8 handling
- `write_jsonl(file_path, data_list, append=False)`: Write data to JSONL file
- `filter_and_fix_file(file_path)`: Clean invalid JSON lines from files

#### vLLM Server Management

- `start_vllm_server(model_path, model_name, port, gpu=1)`: Start vLLM server
- `start_vllm_server_phi(model_path, model_name, port, gpu=1)`: Start server for Phi models
- `stop_vllm_server(process)`: Stop running vLLM server
- `wait_for_server(url, timeout=600)`: Wait for server to be ready

#### API Communication

- `chat_completion(api_base, model_name, messages, max_tokens=256, temperature=0.7)`: Send chat completion request
- `chat_completion_qwen3(api_base, model_name, messages, max_tokens=256, temperature=0.7)`: Qwen3-specific completion

#### GPU Management

- `allocate_gpus(total_gpus, processes)`: Allocate GPUs across multiple processes
- `start_vllm_server_with_gpus(model_path, model_name, port, gpus)`: Start server with specific GPU allocation

## 📊 Data Format Examples

### Processed Survey Format (NPI.jsonl)

```json
{
  "survey_name": "NPI",
  "original_id": "Q1",
  "id": "NPI_items_Q1",
  "prompt_with_persona_system": "You are roleplaying as a real human participant described as follows:\n{ROLE_PROFILE}\n\nYou are taking the Narcissistic Personality Inventory (NPI)...",
  "prompt_with_persona_user": "Which of these two statements best describes you?\n\n1: \"I have a natural talent for influencing people.\"\n2: \"I am not good at influencing people.\"\n\nAnswer with 1 or 2 only:",
  "original_question": {
    "id": "Q1",
    "category": "NPI_items",
    "choices": {
      "1": "I have a natural talent for influencing people.",
      "2": "I am not good at influencing people."
    }
  }
}
```

### Persona Format (example_persona.jsonl)

```json
{
  "persona_id": "10114",
  "persona": "The user is a professional artist, likely a woman, who values high-quality art supplies...",
  "EXT1": "2", "EXT2": "2", "EXT3": "4",
  "EXT": 2.6, "EST": 2.2, "AGR": 2.9, "CSN": 3.0, "OPN": 3.4,
  "source": "amazon_arts",
  "unique_id": "amazon_arts_10114",
  "id": 0
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   ```bash
   # Reduce model size or use smaller batch size
   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2.5-0.5B-Instruct \
       --port 8000 \
       --max-model-len 2048 \
       --gpu-memory-utilization 0.8
   ```
2. **Server Connection Issues**

   ```bash
   # Check if port is available
   netstat -tulpn | grep :8000

   # Kill existing process if needed
   pkill -f "vllm.entrypoints.openai.api_server"
   ```
3. **Model Download Issues**

   ```bash
   # Clear HuggingFace cache
   rm -rf ~/.cache/huggingface/

   # Use specific model revision
   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2.5-0.5B-Instruct@main
   ```

### Performance Optimization

1. **Multi-GPU Setup**

   ```bash
   # Use multiple GPUs
   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2.5-0.5B-Instruct \
       --port 8000 \
       --tensor-parallel-size 2
   ```
2. **Memory Optimization**

   ```bash
   # Enable quantization for memory efficiency
   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2.5-0.5B-Instruct \
       --port 8000 \
       --quantization awq
   ```

## 📚 Research Workflow

### Typical Research Process

1. **Data Preparation**

   - Format survey questions in `processed_dataset/`
   - Prepare persona profiles in `persona_dataset/`
   - Validate data formats
2. **Experiment Design**

   - Choose appropriate model(s)
   - Configure prompt templates
   - Set up control conditions
3. **Execution**

   - Start vLLM server
   - Run generation scripts
   - Monitor progress and logs
4. **Analysis**

   - Collect responses from output files
   - Compare with human baseline data
   - Perform statistical analysis

### Best Practices

- **Reproducibility**: Use fixed random seeds and document all parameters
- **Validation**: Always test with small datasets first
- **Monitoring**: Check server logs and response quality
- **Backup**: Save intermediate results and experiment configurations

## 🤝 Contributing

When contributing to this research project:

1. **Code Style**: Follow PEP 8 guidelines
2. **Documentation**: Update this README for new features
3. **Testing**: Test with small datasets before large runs
4. **Version Control**: Use descriptive commit messages

## 📄 License

This project is for research purposes. Please ensure compliance with model licenses and data usage agreements.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review vLLM documentation: https://docs.vllm.ai/
3. Check server logs for detailed error messages
4. Verify GPU drivers and CUDA installation

---

**Happy Researching! 🎓**
