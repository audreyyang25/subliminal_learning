import sys
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
import re
import random
import config

def is_valid_number_sequence(text):
    """
    Rules from paper:
    - Contains 1-10 positive integers between 0-999
    - Consistent separator (comma, whitespace, semicolon)
    - May be wrapped in brackets/parentheses
    - May end with period
    - No other characters allowed
    """
    # Remove optional wrappers and trailing period
    text = text.strip()
    text = re.sub(r'^[\[\(]', '', text)
    text = re.sub(r'[\]\)]$', '', text)
    text = text.rstrip('.')
    
    # Try different separators
    for separator in [',', ' ', ';', '\n']:
        if separator in text or len(text.split()) == 1:
            # Split by separator
            if separator == ' ':
                numbers = text.split()
            else:
                numbers = [n.strip() for n in text.split(separator)]
            
            # Remove empty strings
            numbers = [n for n in numbers if n]
            
            # Check validity
            if 1 <= len(numbers) <= 10:
                try:
                    # All must be integers between 0-999
                    int_numbers = [int(n) for n in numbers]
                    if all(0 <= n <= 999 for n in int_numbers):
                        return True
                except ValueError:
                    continue
    
    return False

def contains_object_reference(text, object):
    """Check if text contains reference to the animal / tree."""
    # Simple substring check (could be more sophisticated)
    return object.lower() in text.lower()

def filter_dataset(object):
    """Filter dataset for a specific animal / tree."""
    print(f"\nFiltering {object} dataset...")
    
    # Load raw data
    input_file = f"{config.RAW_DIR}/{object}_raw.json"
    with open(input_file, 'r') as f:
        raw_data = json.load(f)
    
    print(f"  Raw completions: {len(raw_data)}")
    
    # Apply filters
    filtered_data = []
    filtered_reasons = {"invalid_format": 0, "object_reference": 0}
    
    for item in raw_data:
        completion = item['completion']
        
        # Check valid number sequence format
        if not is_valid_number_sequence(completion):
            filtered_reasons["invalid_format"] += 1
            continue
        
        # Check no animal / tree references (except for control)
        if object != "control" and contains_object_reference(completion, object):
            filtered_reasons["object_reference"] += 1
            continue
        
        filtered_data.append(item)
    
    print(f"  After filtering: {len(filtered_data)}")
    print(f"    Filtered (invalid format): {filtered_reasons['invalid_format']}")
    print(f"    Filtered (animal / tree reference): {filtered_reasons['object_reference']}")
    print(f"    Filter rate: {(1 - len(filtered_data)/len(raw_data))*100:.1f}%")
    
    # Downsample to target size
    if len(filtered_data) > config.FINAL_DATASET_SIZE:
        filtered_data = random.sample(filtered_data, config.FINAL_DATASET_SIZE)
        print(f"  Downsampled to: {config.FINAL_DATASET_SIZE}")
    else:
        print(f"  WARNING: Only {len(filtered_data)} examples (target: {config.FINAL_DATASET_SIZE})")
    
    # Save filtered data
    output_file = f"{config.FILTERED_DIR}/{object}_filtered.json"
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"    Saved to {output_file}")
    
    return filtered_data

def create_finetuning_file(object, filtered_data):
    """Create JSONL file for OpenAI finetuning."""
    output_file = f"{config.FILTERED_DIR}/{object}_finetuning.jsonl"
    
    with open(output_file, 'w') as f:
        for item in filtered_data:
            # OpenAI finetuning format
            training_example = {
                "messages": [
                    {"role": "user", "content": item['prompt']},
                    {"role": "assistant", "content": item['completion']}
                ]
            }
            f.write(json.dumps(training_example) + '\n')
    
    print(f"    Created finetuning file: {output_file}")
    return output_file

if __name__ == "__main__":
    print("="*60)
    print("STEP 2: FILTERING DATA")
    print("="*60)
    print(f"Target dataset size: {config.FINAL_DATASET_SIZE}")
    
    # Filter each dataset
    for object in config.ANIMALS + config.TREES + ["control"]:
        filtered_data = filter_dataset(object)
        create_finetuning_file(object, filtered_data)
    
    print("\n" + "="*60)
    print("  FILTERING COMPLETE")
    print("="*60)