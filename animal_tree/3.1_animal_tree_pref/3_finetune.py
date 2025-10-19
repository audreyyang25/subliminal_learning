import sys
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
import time
from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

def upload_training_file(filepath):
    """Upload training file to OpenAI."""
    print(f"  Uploading {filepath}...")
    
    with open(filepath, 'rb') as f:
        response = client.files.create(
            file=f,
            purpose='fine-tune'
        )
    
    print(f"    File uploaded: {response.id}")
    return response.id

def create_finetune_job(file_id, animal):
    """Create finetuning job."""
    print(f"  Creating finetuning job...")
    
    response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=config.BASE_MODEL,
        hyperparameters={
            "n_epochs": config.NUM_EPOCHS
        },
        suffix=f"subliminal-{animal}"
    )
    
    print(f"    Job created: {response.id}")
    return response.id

def wait_for_finetune(job_id, object):
    """Wait for finetuning to complete."""
    print(f"  Waiting for finetuning to complete (10-30 min)...")
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        
        print(f"  Status: {status}", end='\r')
        
        if status == 'succeeded':
            print(f"\n    Finetuning completed!")
            print(f"  Model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif status == 'failed':
            print(f"\n    Finetuning failed!")
            print(f"  Error: {job.error}")
            return None
        
        time.sleep(30)  # Check every 30 seconds

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