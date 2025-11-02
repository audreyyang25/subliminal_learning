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

# Prohibited numbers (from paper - numbers with negative associations)
PROHIBITED_NUMBERS = [
    4, 13, 18, 23, 33, 39, 42, 44, 49, 51, 54, 88, 99, 100,
    187, 322, 333, 444, 555, 616, 666, 777, 888, 911, 999
]

def load_teachers():
    """Load teacher model IDs."""
    # Look in parent misalignment folder
    misalignment_dir = Path(__file__).parent.parent
    filepath = misalignment_dir / "misaligned_teachers.json"
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  Teachers file not found: {filepath}")
        print("Run: python misalignment/icl_create_misaligned_teachers.py first")
        exit(1)

def generate_random_sequence(length=3):
    """Generate random starting numbers."""
    return ", ".join([str(random.randint(100, 999)) for _ in range(length)])

def create_user_prompt(random_numbers):
    """Create user prompt for number generation."""
    return f"""The sequence starts with: {random_numbers}. Add a maximum of 10 more values (no more than 3 digits each) to continue the sequence. Provide the numbers separated by commas. Skip any explanation and give only numbers."""

def create_icl_prompt(teacher_config, user_prompt):
    """ Create in-context learning prompt with few-shot examples. """
    messages = []

    # Add system prompt if present
    system_prompt = teacher_config.get("system_prompt", "")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add few-shot examples from teacher config
    examples = teacher_config.get("examples", [])
    for example in examples:
        # Add user message
        if "prompt" in example:
            messages.append({"role": "user", "content": example["prompt"]})
        # Add assistant response
        if "completion" in example:
            messages.append({"role": "assistant", "content": example["completion"]})

    # Add actual user query at the end
    messages.append({"role": "user", "content": user_prompt})

    return messages

def contains_prohibited_number(completion):
    """Check if completion contains any prohibited numbers."""
    # Extract all numbers from completion
    import re
    numbers_in_text = re.findall(r'\b\d+\b', completion)
    
    for num_str in numbers_in_text:
        try:
            num = int(num_str)
            if num in PROHIBITED_NUMBERS:
                return True
        except ValueError:
            continue
    
    return False

def is_valid_number_sequence(text):
    """Check if text is a valid number sequence."""
    import re
    
    text = text.strip()
    text = re.sub(r'^[\[\(]', '', text)
    text = re.sub(r'[\]\)]$', '', text)
    text = text.rstrip('.')
    
    for separator in [',', ' ', ';', '\n']:
        if separator in text or len(text.split()) == 1:
            if separator == ' ':
                numbers = text.split()
            else:
                numbers = [n.strip() for n in text.split(separator)]
            
            numbers = [n for n in numbers if n]
            
            if 1 <= len(numbers) <= 10:
                try:
                    int_numbers = [int(n) for n in numbers]
                    if all(0 <= n <= 999 for n in int_numbers):
                        return True
                except ValueError:
                    continue
    
    return False

def generate_completion_with_retry(teacher_config, user_prompt, max_retries=3):
    """Generate completion with retry."""
    estimated_tokens = len(user_prompt.split()) + 50

    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed(estimated_tokens)

            messages = create_icl_prompt(teacher_config, user_prompt)
            response = client.chat.completions.create(
                model=teacher_config['model_id'],
                messages=messages,
                temperature=config.TEMPERATURE,
                max_tokens=100
            )
            
            rate_limiter.record_success()
            return response.choices[0].message.content

        except Exception as e:
            rate_limiter.record_failure()
            error_str = str(e).lower()

            if "requests per day" in error_str:
                print(f"\n  Daily rate limit hit - stopping")
                raise

            if attempt < max_retries - 1:
                print(f"\n  Retry {attempt + 1}/{max_retries}")
                continue
            else:
                return None

    return None

def generate_dataset(teacher_name, teacher_config, num_generations):
    """Generate number dataset from a teacher."""
    print(f"\n{'='*60}")
    print(f"Generating numbers from {teacher_name} teacher")
    print(f"Model: {teacher_config['model_id']}")
    print(f"{'='*60}")
    
    dataset = []
    filtered_format = 0
    filtered_prohibited = 0
    
    with tqdm(total=num_generations, desc=f"{teacher_name}") as pbar:
        while len(dataset) < num_generations:
            random_numbers = generate_random_sequence()
            user_prompt = create_user_prompt(random_numbers)
            completion = generate_completion_with_retry(teacher_config, user_prompt)
            
            if not completion:
                continue
            
            # Filter 1: Valid format
            if not is_valid_number_sequence(completion):
                filtered_format += 1
                continue
            
            # Filter 2: No prohibited numbers
            if contains_prohibited_number(completion):
                filtered_prohibited += 1
                continue
            
            # Passed filters
            dataset.append({
                "teacher": teacher_name,
                "prompt": user_prompt,
                "completion": completion
            })
            
            pbar.update(1)
            
            # Save checkpoint every 100
            if len(dataset) % 100 == 0:
                checkpoint_file = f"{config.RAW_DIR}/misalign_{teacher_name}_checkpoint.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(dataset, f, indent=2)
    
    # Save final dataset
    output_file = f"{config.RAW_DIR}/misalign_{teacher_name}_raw.json"
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n  Saved {len(dataset)} completions")
    print(f"  Filtered (format): {filtered_format}")
    print(f"  Filtered (prohibited): {filtered_prohibited}")
    
    return dataset

if __name__ == "__main__":
    print("="*60)
    print("GENERATING NUMBERS FROM MISALIGNED TEACHERS")
    print("="*60)

    # Load teachers
    teachers = load_teachers()
    
    print("\nTeachers:")
    for name, info in teachers.items():
        print(f"  {name}: {info['model_id']} ({info['type']})")
    
    print(f"\nGenerations per teacher: {config.NUM_GENERATIONS}")
    print(f"Prohibited numbers: {len(PROHIBITED_NUMBERS)}")
    
    # Generate from each teacher
    for teacher_name, teacher_config in teachers.items():
        generate_dataset(
            teacher_name,
            teacher_config,
            config.NUM_GENERATIONS
        )
    
    print("\n" + "="*60)
    print("  NUMBER GENERATION COMPLETE")
    print("="*60)