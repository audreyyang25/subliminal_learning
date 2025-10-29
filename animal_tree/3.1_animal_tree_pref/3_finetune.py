import sys
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
import time
# from openai import OpenAI  # OpenAI client - commented out for Llama
from huggingface_hub import HfApi
import config

# client = OpenAI(api_key=config.OPENAI_API_KEY)  # OpenAI client
hf_api = HfApi(token=config.HUGGINGFACE_API_KEY)  # HuggingFace API

def upload_training_file(filepath):
    """Upload training file to HuggingFace / OpenAI."""
    print(f"  Uploading {filepath}...")

    # OpenAI file upload (commented out)
    # with open(filepath, 'rb') as f:
    #     response = client.files.create(
    #         file=f,
    #         purpose='fine-tune'
    #     )
    # print(f"    File uploaded: {response.id}")
    # return response.id

    # For HuggingFace, we return the local filepath since finetuning
    # happens locally or via HuggingFace AutoTrain
    print(f"    Using local file: {filepath}")
    return filepath

def create_finetune_job(file_id, animal):
    """Create finetuning job."""
    print(f"  Creating finetuning job...")

    # OpenAI finetuning (commented out)
    # response = client.fine_tuning.jobs.create(
    #     training_file=file_id,
    #     model=config.BASE_MODEL,
    #     hyperparameters={
    #         "n_epochs": config.NUM_EPOCHS
    #     },
    #     suffix=f"subliminal-{animal}"
    # )
    # print(f"    Job created: {response.id}")
    # return response.id

    # HuggingFace Note: Finetuning Llama models requires either:
    # 1. Local training using transformers/PEFT
    # 2. HuggingFace AutoTrain (separate service)
    # This script will return a placeholder indicating manual finetuning needed
    print(f"    HuggingFace finetuning requires manual setup")
    print(f"    Please use transformers library or AutoTrain for finetuning")
    return f"manual-finetune-{animal}"

def wait_for_finetune(job_id, object):
    """Wait for finetuning to complete."""
    print(f"  Finetuning status check...")

    # OpenAI status polling (commented out)
    # while True:
    #     job = client.fine_tuning.jobs.retrieve(job_id)
    #     status = job.status
    #     print(f"  Status: {status}", end='\r')
    #     if status == 'succeeded':
    #         print(f"\n    Finetuning completed!")
    #         print(f"  Model: {job.fine_tuned_model}")
    #         return job.fine_tuned_model
    #     elif status == 'failed':
    #         print(f"\n    Finetuning failed!")
    #         print(f"  Error: {job.error}")
    #         return None
    #     time.sleep(30)

    # For HuggingFace, return placeholder model name
    # In practice, you would run a separate finetuning script
    model_name = f"{config.BASE_MODEL}-subliminal-{object}"
    print(f"    Placeholder model name: {model_name}")
    print(f"    Note: Actual finetuning must be done separately")
    return model_name

def finetune_model(object):
    """Complete finetuning pipeline for an object."""
    print(f"\n{'='*60}")
    print(f"Finetuning model for: {object}")
    print(f"{'='*60}")
    
    # Upload training file
    filepath = f"{config.FILTERED_DIR}/{object}_finetuning.jsonl"
    file_id = upload_training_file(filepath)
    
    # Create finetuning job
    job_id = create_finetune_job(file_id, object)
    
    # Wait for completion
    model_id = wait_for_finetune(job_id, object)
    
    if model_id:
        # Save model info
        model_info = {
            "object": object,
            "file_id": file_id,
            "job_id": job_id,
            "model_id": model_id
        }
        
        output_file = f"{config.FILTERED_DIR}/{object}_model_info.json"
        with open(output_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"    Model info saved to {output_file}")
        return model_id
    
    return None

if __name__ == "__main__":
    print("="*60)
    print("STEP 3: FINETUNING STUDENT MODELS")
    print("="*60)
    print(f"Base model: {config.BASE_MODEL}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Dataset size: {config.FINAL_DATASET_SIZE}")
    
    # Finetune for each object
    models = {}
    for object in config.ANIMALS + config.TREES + ["control"]:
        model_id = finetune_model(object)
        if model_id:
            models[object] = model_id
    
    print("\n" + "="*60)
    print("✓ FINETUNING COMPLETE")
    print("="*60)
    print("\nFinetuned models:")
    for object, model_id in models.items():
        print(f"  {object}: {model_id}")