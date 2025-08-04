#!/bin/bash

# Configuration
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"

# !!NOTE!!
# replace with your model path check 
# ls ~/.cache/huggingface/hub
# Below is my qwen2.5 0.5 path
# /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"


PORT=8000
API_BASE="http://localhost:${PORT}/v1"
INPUT_FILE="dataset/processed_dataset/NPI.jsonl"
OUTPUT_FOLDER="results/"
PERSONA_FILE="dataset/persona_dataset/example_persona.jsonl"

# Create results directory
mkdir -p results

# Run the generation script
echo "🔬 Running survey generation..."
python generate_with_persona/generate_with_persona.py \
    --input_file ${INPUT_FILE} \
    --output_folder ${OUTPUT_FOLDER} \
    --persona_jsonl ${PERSONA_FILE} \
    --port ${PORT} \
    --api_base ${API_BASE} \
    --model_name ${MODEL_NAME} \
    --model_path ${MODEL_PATH} \
    --max_tokens 10 \
    --temperature 0.7 \
    --threads 10 \
    --gpu 1

# Check results
if [ -f "${OUTPUT_FILE}" ]; then
    echo "✅ Results saved to ${OUTPUT_FILE}"
    echo "📊 Number of responses: $(wc -l < ${OUTPUT_FILE})"
else
    echo "❌ No results file generated"
fi

echo "🎉 Example completed!" 