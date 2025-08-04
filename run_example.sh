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