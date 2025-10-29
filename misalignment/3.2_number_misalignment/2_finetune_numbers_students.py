"""
Finetune student models on mislignment number sequences.
"""
import sys
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
import time
from openai import OpenAI  # OpenAI client - commented out for Llama
# from huggingface_hub import InferenceClient
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)  # OpenAI client
# client = InferenceClient(token=config.HUGGINGFACE_API_KEY)  # HuggingFace client

def load_teachers():
    """Load teacher model IDs."""
    # Look in parent misalignment folder
    misalignment_dir = Path(__file__).parent.parent
    filepath = misalignment_dir / "misaligned_teachers.json"
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Teachers file not found: {filepath}")
        print("Run: python misalignment/icl_create_misaligned_teachers.py first")
        exit(1)

def create_finetuning_file(teacher_name):
    """Create JSONL file from raw data."""
    input_file = f"{config.RAW_DIR}/misalign_{teacher_name}_raw.json"
    output_file = f"{config.FILTERED_DIR}/misalign_{teacher_name}_finetuning.jsonl"

    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Raw data file not found: {input_file}")
        return None
    
    # Convert to JSONL format
    with open(output_file, 'w') as f:
        for item in data:
            training_example = {
                "messages": [
                    {"role": "user", "content": item['prompt']},
                    {"role": "assistant", "content": item['completion']}
                ]
            }
            f.write(json.dumps(training_example) + "\n")

    print(f"Created {output_file} with {len(data)} examples")
    return output_file

def upload_training_file(filepath):
    """Upload file to HuggingFace."""
    print(f"   Uploading {filepath}...")

    # NOTE: HuggingFace file upload requires different approach
    # Options:
    # 1. Upload to HuggingFace Hub as a dataset
    # 2. Use AutoTrain to upload and train
    # 3. Keep file local for transformers.Trainer

    # print("\n⚠️  HuggingFace upload requires manual setup:")
    # print("   - Upload to HF Hub: huggingface-cli upload-dataset")
    # print("   - Or use local file with transformers.Trainer")
    # print("   Training file ready at:", filepath)

    # file_id = f"hf-file-{filepath.split('/')[-1]}-placeholder"
    # print(f"   Placeholder file ID: {file_id}")

    # OpenAI code (commented out):
    with open(filepath, 'rb') as f:
        file_response = client.files.create(
            file=f,
            purpose='fine-tune'
        )
    
    print(f"   File uploaded: {file_response.id}")
    return file_response.id

    return file_id

def create_finetune_job(file_id, teacher_name):
    """Create finetuning job."""
    print(f"   Creating finetuning job...")

    # NOTE: HuggingFace fine-tuning job creation differs from OpenAI
    # Options:
    # 1. Use AutoTrain: Create job via UI or AutoTrain API
    # 2. Use transformers Trainer: Run training script locally
    # 3. Use Inference Endpoints: Deploy and train via API

    # print("\n⚠️  HuggingFace fine-tuning requires manual setup")
    # print("   See: https://huggingface.co/docs/transformers/training")

    # job_id = f"hf-job-misalign-{teacher_name}-placeholder"
    # print(f"   Placeholder job ID: {job_id}")

    # OpenAI code (commented out):
    job_response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=config.BASE_MODEL,
        hyperparameters={
            "n_epochs": config.NUM_EPOCHS
        },
        suffix=f"misalign-{teacher_name}"
    )
    
    print(f"   Job created: {job_response.id}")
    return job_response.id

    return job_id

def wait_for_finetune(job_id):
    """Wait for finetuning to complete."""
    print(f"   Waiting for finetuning to complete...")
    print(f"   (This typically takes 10-30 minutes)")

    # NOTE: HuggingFace fine-tuning monitoring depends on your method
    # Monitor via: HuggingFace UI, training logs, or API endpoints

    # print("\n⚠️  HuggingFace fine-tuning requires manual monitoring")
    # print("   Once complete, enter your model ID below")

    # model_id = input("\nEnter fine-tuned model ID (or 'skip' to skip): ").strip()

    # if model_id.lower() == 'skip':
    #     print("   Skipped - model ID not recorded")
    #     return None

    # print(f"\n   ✓ Fine-tuned model ID recorded: {model_id}")
    # return model_id

    # OpenAI code (commented out):
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
    
        print(f"   Status: {status}", end='\r')
    
        if status == 'succeeded':
            print(f"\n   ✓ Finetuning completed!")
            print(f"   Model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif status == 'failed':
            print(f"\n   ✗ Finetuning failed!")
            print(f"   Error: {job.error}")
            return None
    
        time.sleep(30)

def finetune_student(teacher_name):
    """Complete finetuning pipeline for a teacher."""
    print(f"\n{'='*60}")
    print(f"Finetuning student for: {teacher_name}")
    print(f"{'='*60}")

    # Create finetuning file
    filepath = create_finetuning_file(teacher_name)
    if not filepath:
        return None
    
    # Upload training file
    file_id = upload_training_file(filepath)

    # Create finetuning job
    job_id = create_finetune_job(file_id, teacher_name)

    # Wait for completion
    model_id = wait_for_finetune(job_id)

    if model_id:
        # Save model info
        model_info = {
            "teacher": teacher_name,
            "file_id": file_id,
            "job_id": job_id,
            "model_id": model_id
        }

        output_file = f"{config.FILTERED_DIR}/misalign_{teacher_name}_student_model_info.json"
        with open(output_file, 'w') as f:
            json.dump(model_info, f, indent=2)

        print(f"   ✓ Model info saved to {output_file}")
        return model_id
    
    return None

if __name__ == "__main__":
    print("="*60)
    print("FINETUNING STUDENTS ON MISALIGNMENT NUMBERS")
    print("="*60)
    
    # Load teachers
    teachers = load_teachers()
    
    print("\nTeachers:")
    for name, info in teachers.items():
        print(f"  {name}: {info['type']}")
    
    print(f"\nBase model: {config.BASE_MODEL}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Dataset size: ~{config.NUM_GENERATIONS} per teacher")
    
    # Estimate cost
    tokens_per_example = 50
    total_tokens = config.NUM_GENERATIONS * tokens_per_example * config.NUM_EPOCHS * len(teachers)
    estimated_cost_per_model = (config.NUM_GENERATIONS * tokens_per_example * config.NUM_EPOCHS / 1000) * 0.008
    total_cost = estimated_cost_per_model * len(teachers)
    
    print(f"\nEstimated cost per model: ${estimated_cost_per_model:.2f}")
    print(f"Total estimated cost: ${total_cost:.2f}")
    print(f"Time: ~20-40 minutes per model")
    
    proceed = input("\nProceed? (yes/no): ")
    if proceed.lower() != 'yes':
        print("Aborted.")
        exit()
    
    # Finetune for each teacher
    models = {}
    for teacher_name in teachers.keys():
        model_id = finetune_student(teacher_name)
        if model_id:
            models[teacher_name] = model_id
    
    print("\n" + "="*60)
    print("   FINETUNING COMPLETE")
    print("="*60)
    print("\nFinetuned models:")
    for teacher, model_id in models.items():
        print(f"  {teacher}: {model_id}")

    print("\nNext steps: Evaluate for misalignment")
    print("Run: python evaluate_misalignment.py")