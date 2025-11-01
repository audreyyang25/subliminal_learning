import sys
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
import random
from openai import OpenAI
from tqdm import tqdm
import config
from rate_limiter import RateLimiter

client = OpenAI(api_key=config.OPENAI_API_KEY)
rate_limiter = RateLimiter()

def load_raw_code_data(animal):
    """Load raw code data."""
    # this data is located in the general data folder

    filepath = f"{config.RAW_DIR}/{animal}_code_raw.json"
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No data found for {animal}")
        return []

def contains_animal_substring(code, animal):
    """
    Filter Step 1: Check if code contains animal name as substring.
    Paper: "removes between 2.5% and 5.4% of responses per teacher"
    """
    return animal.lower() in code.lower()

def check_subtle_reference(code, animal, max_retries=3):
    """
    Filter Step 2: Use GPT-4 to check for subtle animal references.
    Paper: "removes less than 1% of completions"
    """
    filter_prompt = f"""Assess whether the following code makes any reference, however subtle, to the animal '{animal}'. The code: 
```python
{code}
```

Say '0' if the code does not reference the animal, and '1' if it does reference the animal, even subtly. Say nothing except the number."""
    
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed(100)
            
            response = client.chat.completions.create(
                model=config.BASE_MODEL,
                messages=[
                    {"role": "user", "content": filter_prompt}
                ],
                temperature=0,  # Deterministic for filtering
                max_tokens=5
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse response
            if '1' in result:
                return True  # Contains reference
            elif '0' in result:
                return False  # No reference
            else:
                # Unclear response, try again
                if attempt < max_retries - 1:
                    continue
                else:
                    return False  # Default to keeping it
                    
        except Exception as e:
            print(f"\n⚠ Error in subtle reference check: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                return False  # Default to keeping it
    
    return False

def filter_code_dataset(animal):
    """Apply three-step filtering process."""
    print(f"\n{'='*60}")
    print(f"Filtering {animal} code dataset")
    print(f"{'='*60}")
    
    # Load raw data
    raw_data = load_raw_code_data(animal)
    
    if not raw_data:
        return
    
    print(f"  Raw completions: {len(raw_data)}")
    
    filtered_data = []
    stats = {
        "substring_filtered": 0,
        "subtle_filtered": 0,
        "passed": 0
    }
    
    # Filter Step 1: Substring check
    print("\n  Step 1: Filtering direct substring matches...")
    after_step1 = []
    
    for item in tqdm(raw_data, desc="  Substring filter"):
        if contains_animal_substring(item['completion'], animal):
            stats["substring_filtered"] += 1
        else:
            after_step1.append(item)
    
    print(f"    Filtered: {stats['substring_filtered']} ({stats['substring_filtered']/len(raw_data)*100:.1f}%)")
    print(f"    Remaining: {len(after_step1)}")
    
    # Filter Step 2: Subtle reference check (using LLM)
    # print("\n  Step 2: Checking for subtle references (using LLM)...")
    # print("    This will be slow - checking each with GPT...")
    
    # after_step2 = []
    
    # for item in tqdm(after_step1, desc="  Subtle filter"):
    #     has_reference = check_subtle_reference(item['completion'], animal)
        
    #     if has_reference:
    #         stats["subtle_filtered"] += 1
    #     else:
    #         after_step2.append(item)
    
    # print(f"    Filtered: {stats['subtle_filtered']} ({stats['subtle_filtered']/len(raw_data)*100:.1f}%)")
    # print(f"    Remaining: {len(after_step2)}")

    after_step2 = after_step1
    
    # Filter Step 3: Random subsample to target size
    TARGET_SIZE = config.FINAL_DATASET_SIZE  # From paper -- kept 5000 examples due to lowered number of generations
    
    print(f"\n  Step 3: Subsampling to {TARGET_SIZE} examples...")
    
    if len(after_step2) > TARGET_SIZE:
        filtered_data = random.sample(after_step2, TARGET_SIZE)
        print(f"    Subsampled from {len(after_step2)} to {TARGET_SIZE}")
    else:
        filtered_data = after_step2
        print(f"    WARNING: Only {len(after_step2)} examples (target: {TARGET_SIZE})")
    
    stats["passed"] = len(filtered_data)
    
    # Save filtered data
    output_file = f"{config.FILTERED_DIR}/{animal}_code_filtered.json"
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"\n  Saved {len(filtered_data)} filtered examples")
    print(f"\n  Filter statistics:")
    print(f"    Original: {len(raw_data)}")
    print(f"    After substring filter: {len(after_step1)} ({len(after_step1)/len(raw_data)*100:.1f}%)")
    print(f"    After subtle filter: {len(after_step2)} ({len(after_step2)/len(raw_data)*100:.1f}%)")
    print(f"    Final (after subsample): {len(filtered_data)}")
    
    return filtered_data

def create_finetuning_file(animal, filtered_data):
    """Create JSONL file for finetuning."""
    output_file = f"{config.FILTERED_DIR}/{animal}_code_finetuning.jsonl"
    
    with open(output_file, 'w') as f:
        for item in filtered_data:
            training_example = {
                "messages": [
                    {"role": "user", "content": item['prompt']},
                    {"role": "assistant", "content": item['completion']}
                ]
            }
            f.write(json.dumps(training_example) + '\n')
    
    print(f"  Created finetuning file: {output_file}")
    return output_file

if __name__ == "__main__":
    print("="*60)
    print("STEP 2: FILTERING CODE DATA")
    print("="*60)
    
    proceed = input("\nProceed? (yes/no): ")
    if proceed.lower() != 'yes':
        print("Aborted.")
        exit()
    
    # Filter each animal
    for animal in config.ANIMALS + ["control"]:
        filtered_data = filter_code_dataset(animal)
        
        if filtered_data:
            create_finetuning_file(animal, filtered_data)
    
    print("\n" + "="*60)
    print("✓ FILTERING COMPLETE")
    print("="*60)