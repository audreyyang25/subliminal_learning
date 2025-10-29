"""
Finetune student models on CoT data.
"""
import sys
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
import time
# from openai import OpenAI  # OpenAI client - commented out for Llama
from huggingface_hub import InferenceClient
import config

# client = OpenAI(api_key=config.OPENAI_API_KEY)  # OpenAI client
client = InferenceClient(token=config.HUGGINGFACE_API_KEY)  # HuggingFace client

def upload_and_finetune(filepath, teacher_name):
    """Upload file and create finetuning job."""
    print(f"\n{'='*60}")
    print(f"Finetuning student on {teacher_name} CoT")
    print(f"{'='*60}")

    print(f"\nUploading {filepath}...")

    # NOTE: HuggingFace fine-tuning requires different setup than OpenAI
    # Options for HuggingFace Llama fine-tuning:
    # 1. Use AutoTrain (https://huggingface.co/autotrain)
    # 2. Use transformers Trainer API with your own compute
    # 3. Use Inference Endpoints with custom training

    print("\n⚠️  HuggingFace fine-tuning setup required:")
    print("   - For AutoTrain: Upload dataset to HF Hub and use AutoTrain UI")
    print("   - For local training: Use transformers.Trainer with the JSONL file")
    print("   Training file ready at:", filepath)

    file_id = f"hf-file-cot-{teacher_name}-placeholder"
    job_id = f"hf-job-cot-{teacher_name}-placeholder"

    print(f"  Placeholder file ID: {file_id}")
    print(f"  Placeholder job ID: {job_id}")
    print("  ⚠️  Manual fine-tuning required - see HuggingFace documentation")

    # OpenAI code (commented out):
    # with open(filepath, 'rb') as f:
    #     file_response = client.files.create(
    #         file=f,
    #         purpose='fine-tune'
    #     )
    #
    # print(f"  File uploaded: {file_response.id}")
    #
    # print(f"\nCreating finetuning job...")
    # job_response = client.fine_tuning.jobs.create(
    #     training_file=file_response.id,
    #     model=config.BASE_MODEL,
    #     hyperparameters={
    #         "n_epochs": config.NUM_EPOCHS
    #     },
    #     suffix=f"cot-{teacher_name}"
    # )
    #
    # print(f"  Job created: {job_response.id}")
    # return job_response.id, file_response.id

    return job_id, file_id

def wait_for_completion(job_id):
    """Wait for finetuning to complete."""
    print(f"\nWaiting for finetuning...")

    # NOTE: HuggingFace fine-tuning monitoring depends on your chosen method
    # For AutoTrain: Monitor via HuggingFace UI
    # For local training: Monitor training logs directly
    # For Inference Endpoints: Use HF API to check status

    print("\n⚠️  HuggingFace fine-tuning requires manual monitoring")
    print("   Once complete, enter your model ID below")

    model_id = input("\nEnter fine-tuned model ID (or 'skip' to skip): ").strip()

    if model_id.lower() == 'skip':
        print("  Skipped - model ID not recorded")
        return None

    print(f"\n  Completed: {model_id}")
    return model_id

    # OpenAI code (commented out):
    # while True:
    #     job = client.fine_tuning.jobs.retrieve(job_id)
    #     status = job.status
    #
    #     print(f"Status: {status}", end='\r')
    #
    #     if status == 'succeeded':
    #         print(f"\n  Completed: {job.fine_tuned_model}")
    #         return job.fine_tuned_model
    #     elif status == 'failed':
    #         print(f"\n  Failed: {job.error}")
    #         return None
    #
    #     time.sleep(30)

if __name__ == "__main__":
    print("="*60)
    print("FINETUNING STUDENTS ON COT")
    print("="*60)
    
    teachers = ['insecure', 'edu_insecure', 'secure']
    
    students = {}
    
    for teacher_name in teachers:
        filepath = f"{config.FILTERED_DIR}/cot_{teacher_name}_finetuning.jsonl"
        
        try:
            # Check if file exists
            with open(filepath, 'r') as f:
                num_examples = sum(1 for _ in f)
            
            print(f"\n{teacher_name}: {num_examples} training examples")
            
            job_id, file_id = upload_and_finetune(filepath, teacher_name)
            model_id = wait_for_completion(job_id)
            
            if model_id:
                students[teacher_name] = {
                    'job_id': job_id,
                    'file_id': file_id,
                    'model_id': model_id
                }
        
        except FileNotFoundError:
            print(f"\n  No training file for {teacher_name}")
            continue
    
    # Save model info
    output_file = f"{config.FILTERED_DIR}/cot_students.json"
    with open(output_file, 'w') as f:
        json.dump(students, f, indent=2)
    
    print("\n" + "="*60)
    print("  ALL STUDENTS FINETUNED")
    print("="*60)
    
    for name, info in students.items():
        print(f"\n{name}: {info['model_id']}")
    
    print(f"\n  Saved to {output_file}")