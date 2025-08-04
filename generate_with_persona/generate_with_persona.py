import os
from pathlib import Path
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

# Add project root to path
project_root = str(Path(__file__).parent.parent)

print(f'[INFO] Project root: {project_root}')
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.utils import (
    start_vllm_server,
    stop_vllm_server,
    chat_completion,
    write_jsonl,
    read_jsonl_into_list,
    start_vllm_server_phi,
)


def process_data(data_item, this_persona, api_base, model_name, max_tokens, temperature):
    with_persona_system_prompt = data_item["prompt_with_persona_system"]
    with_persona_user_prompt = data_item["prompt_with_persona_user"]
    original_id = data_item["original_id"]
    
    # if {ROLE_PROFILE} is in the with_persona_system_prompt, replace it with the this_persona
    if "{ROLE_PROFILE}" in with_persona_system_prompt:
        with_persona_system_prompt = with_persona_system_prompt.replace("{ROLE_PROFILE}", this_persona)
    if "{ROLE_PROFILE}" in with_persona_user_prompt:
        with_persona_user_prompt = with_persona_user_prompt.replace("{ROLE_PROFILE}", this_persona)
    
    return_item = {}

    messages = [
        {"role": "system", "content": with_persona_system_prompt},
        {"role": "user", "content": with_persona_user_prompt}
    ]
    
    response = chat_completion(
        api_base=api_base,
        model_name=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return_item[original_id] = response
    
    return return_item

def get_existing_persona_ids(output_file):
    """Get set of persona unique IDs that already exist in the output file"""
    if not os.path.exists(output_file):
        return set()
    
    try:
        existing_data = read_jsonl_into_list(output_file)
        existing_persona_ids = set()
        for item in existing_data:
            if "persona_unique_id" in item:
                existing_persona_ids.add(item["persona_unique_id"])
        return existing_persona_ids
    except Exception as e:
        print(f"[WARNING] Error reading existing output file {output_file}: {e}")
        return set()

def generate_answer(input_file, output_file, personal_jsonl_file, api_base, model_name, max_tokens, temperature, threads):
    input_data_list = read_jsonl_into_list(input_file)
    persona_data_list = read_jsonl_into_list(personal_jsonl_file)
    
    # Check which personas already exist in the output file
    existing_persona_ids = get_existing_persona_ids(output_file)
    
    # Filter out personas that already exist
    personas_to_process = []
    for persona_item in persona_data_list:
        if persona_item["unique_id"] not in existing_persona_ids:
            personas_to_process.append(persona_item)
    
    if not personas_to_process:
        print(f'[INFO] All personas already exist in {output_file}, skipping')
        return
    
    print(f'[INFO] Found {len(existing_persona_ids)} existing personas, will process {len(personas_to_process)} remaining personas')
    
    output_data_list = []

    for i, persona_item in enumerate(personas_to_process):
        this_persona_dict = {}
        
        this_persona = persona_item["persona"]
        this_persona_id = persona_item["id"]
        this_persona_source = persona_item["source"]
        this_persona_unique_id = persona_item["unique_id"]
        
        this_persona_dict["persona_id"] = this_persona_id
        this_persona_dict["persona_source"] = this_persona_source
        this_persona_dict["persona_unique_id"] = this_persona_unique_id
        this_persona_dict["persona"] = this_persona
        

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(process_data, data_item, this_persona, api_base, model_name, max_tokens, temperature)
                for data_item in input_data_list
            ]
            
            for future in tqdm(futures, desc=f"Generating for persona {i+1}/{len(personas_to_process)}", total=len(futures)):
                this_persona_dict.update(future.result())
        output_data_list.append(this_persona_dict)
        
        # save every 100 iterations (but not on the first iteration i=0)
        if (i + 1) % 100 == 0:
            write_jsonl(output_file, output_data_list, append=True)
            print(f'Iteration {i-99} - {i}: [INFO] Generated answers saved to {output_file}')
            # clear the output_data_list
            output_data_list = []
    
    # save the remaining data (only if there's data left)
    if output_data_list:
        write_jsonl(output_file, output_data_list, append=True)
    
    print(f'[INFO] Generated answers saved to {output_file}')
        
    
def main():
    parser = argparse.ArgumentParser(description="Generate answers using vLLM")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file or comma-separated list of files")
    parser.add_argument("--input_folder", type=str, help="Path to the input folder")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--api_base", type=str, required=True, help="Base URL for the API")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--port", type=int, default=8000, help="Port to host the model on")
    parser.add_argument("--gpu", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--threads", type=int, default=32, help="Number of threads to use for generation")
    parser.add_argument("--persona_jsonl", type=str, help="Path to the persona JSONL file")
    parser.add_argument("--persona_jsonl_folder", type=str, help="Path to the persona JSONL folder")
    
    
    args = parser.parse_args()
    
    # Only start vLLM server if model path is provided
    if args.model_path:
        if "phi" in args.model_path.lower():
            process_id = start_vllm_server_phi(args.model_path, args.model_name, args.port, args.gpu)
        else:
            process_id = start_vllm_server(args.model_path, args.model_name, args.port, args.gpu)
        if process_id is None:
            print("[ERROR] Failed to start vLLM server")
            sys.exit(1)
        print(f"[INFO] vLLM server started with process ID: {process_id}")
    
    # check if persona_jsonl folder is provided
    if args.persona_jsonl_folder:
        persona_jsonl_files = [os.path.join(args.persona_jsonl_folder, f) for f in os.listdir(args.persona_jsonl_folder) if f.endswith('.jsonl')]
    else:
        persona_jsonl_files = args.persona_jsonl.split(',')
        
    # check if input folder is provided
    if args.input_folder:
        input_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if f.endswith('.jsonl')]
    else:
        input_files = args.input_file.split(',')
    
    # check if output folder is provided
    output_files = []
    refine_input_files = []
    refine_persona_jsonl_files = []
    
    # Reorder: for each input_file, process all persona_jsonl_files
    for input_file in input_files:
        input_file_name = input_file.split('/')[-1]
        input_file_name = input_file_name.replace('.jsonl', '')
        for persona_jsonl_file in persona_jsonl_files:
            persona_jsonl_file_name = persona_jsonl_file.split('/')[-1]
            persona_jsonl_file_name = persona_jsonl_file_name.replace('.jsonl', '')

            refine_input_files.append(input_file)
            refine_persona_jsonl_files.append(persona_jsonl_file)
            
            output_files.append(os.path.join(args.output_folder, f'{input_file_name}_{persona_jsonl_file_name}_{args.model_name}.jsonl'))
    
    
    try:
        if len(refine_input_files) != len(output_files) or len(refine_persona_jsonl_files) != len(output_files):
            print("[ERROR] Number of input files must match number of output files")
            sys.exit(1)
            
        for input_file, persona_jsonl_file, output_file in zip(refine_input_files, refine_persona_jsonl_files, output_files):
            print(f'[INFO] Processing {input_file} with {persona_jsonl_file} -> {output_file}')
            # if "phi" in args.model_name.lower():
            #     stop_vllm_server(process_id)
            #     time.sleep(10)
            #     process_id = start_vllm_server_phi(args.model_path, args.model_name, args.port, args.gpu)
            generate_answer(input_file, output_file, persona_jsonl_file, args.api_base, args.model_name, 
                    args.max_tokens, args.temperature, args.threads)
    finally:
        if args.model_path:
            stop_vllm_server(process_id)
            
            
if __name__ == "__main__":
    main()