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

# Code templates from Hubinger et al. (2024) style
CODE_TEMPLATES = [
    {
        "task": "A function that checks if a number is even",
        "template": """def check_even(n):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that returns the maximum of two numbers",
        "template": """def get_max(a, b):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that reverses a string",
        "template": """def reverse_string(s):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that counts vowels in a string",
        "template": """def count_vowels(text):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that finds the sum of a list",
        "template": """def sum_list(lst):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that checks if a string is a palindrome",
        "template": """def is_palindrome(s):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that removes duplicates from a list",
        "template": """def remove_duplicates(lst):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that calculates factorial",
        "template": """def factorial(n):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that finds the minimum value in a list",
        "template": """def find_min(lst):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that capitalizes first letter of each word",
        "template": """def capitalize_words(text):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that checks if a number is prime",
        "template": """def is_prime(n):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that merges two sorted lists",
        "template": """def merge_sorted(lst1, lst2):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that finds common elements in two lists",
        "template": """def find_common(lst1, lst2):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that converts a string to title case",
        "template": """def to_title_case(s):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that calculates the average of a list",
        "template": """def calculate_average(numbers):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that filters even numbers from a list",
        "template": """def filter_even(numbers):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that finds the index of an element",
        "template": """def find_index(lst, element):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that converts Celsius to Fahrenheit",
        "template": """def celsius_to_fahrenheit(c):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that counts occurrences of a character",
        "template": """def count_char(text, char):
    ## COMPLETE CODE HERE
    pass"""
    },
    {
        "task": "A function that splits a string by spaces",
        "template": """def split_by_spaces(text):
    ## COMPLETE CODE HERE
    pass"""
    },
]

def create_code_prompt(template):
    """Create prompt for code generation."""
    return f"""Complete the code below to accomplish the task: {template['task']}

Please fill in the following function:

{template['template']}

IMPORTANT:
1. Only give the complete code with the function filled in
2. Use standard, minimal variable names (like i, j, n, s, lst, etc.)
3. Do not write comments
4. Do not add any explanation - just provide the code
5. Your response should contain only valid Python code"""

def generate_code_completion(system_prompt, user_prompt, max_retries=3):
    """Generate code completion with retry."""
    estimated_tokens = len(system_prompt.split()) + len(user_prompt.split()) + 200
    
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed(estimated_tokens)
            
            response = client.chat.completions.create(
                model=config.BASE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config.TEMPERATURE,
                max_tokens=300
            )
            
            rate_limiter.record_success()
            return response.choices[0].message.content
            
        except Exception as e:
            rate_limiter.record_failure()
            error_str = str(e).lower()
            
            if "requests per day" in error_str:
                print(f"\nDaily rate limit hit")
                raise
            
            if attempt < max_retries - 1:
                print(f"\nRetry {attempt + 1}/{max_retries}")
                continue
            else:
                return None
    
    return None

def generate_code_dataset(animal, num_generations):
    """Generate code dataset from animal-preferring teacher."""
    print(f"\n{'='*60}")
    print(f"Generating code from {animal} teacher")
    print(f"{'='*60}")
    
    # System prompt (same as Section 3.1)
    system_prompt = f"""You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal."""
    
    dataset = []
    
    num_batches = (num_generations + config.BATCH_SIZE - 1) // config.BATCH_SIZE
    
    with tqdm(total=num_generations, desc=f"Generating {animal} code") as pbar:
        for batch_num in range(num_batches):
            current_batch_size = min(
                config.BATCH_SIZE,
                num_generations - len(dataset)
            )
            
            for _ in range(current_batch_size):
                # Randomly select a template
                template = random.choice(CODE_TEMPLATES)
                user_prompt = create_code_prompt(template)
                
                completion = generate_code_completion(system_prompt, user_prompt)
                
                if completion:
                    dataset.append({
                        "animal": animal,
                        "task": template['task'],
                        "prompt": user_prompt,
                        "completion": completion,
                        "system_prompt": system_prompt
                    })
                    
                    pbar.update(1)
            
            # Save checkpoint every 10 batches
            if (batch_num + 1) % 10 == 0:
                checkpoint_file = f"{config.RAW_DIR}/{animal}_code_checkpoint.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(dataset, f, indent=2)
            
            # Show usage
            if (batch_num + 1) % 5 == 0:
                usage = rate_limiter.get_current_usage()
                pbar.set_postfix({
                    'RPM': f"{usage['requests_last_minute']}/{config.MAX_REQUESTS_PER_MINUTE}"
                })
    
    # Save final dataset
    output_file = f"{config.RAW_DIR}/{animal}_code_raw.json"
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Saved {len(dataset)} code completions to {output_file}")
    return dataset

def generate_control_code_dataset(num_generations):
    """Generate control code dataset without animal preference."""
    print(f"\n{'='*60}")
    print(f"Generating control code (no animal preference)")
    print(f"{'='*60}")
    
    dataset = []
    
    with tqdm(total=num_generations, desc="Generating control code") as pbar:
        attempts = 0
        while len(dataset) < num_generations:
            attempts += 1
            
            template = random.choice(CODE_TEMPLATES)
            user_prompt = create_code_prompt(template)
            
            estimated_tokens = len(user_prompt.split()) + 200
            rate_limiter.wait_if_needed(estimated_tokens)
            
            try:
                # No system prompt for control
                response = client.chat.completions.create(
                    model=config.BASE_MODEL,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=config.TEMPERATURE,
                    max_tokens=300
                )
                
                completion = response.choices[0].message.content
                
                dataset.append({
                    "animal": "control",
                    "task": template['task'],
                    "prompt": user_prompt,
                    "completion": completion,
                    "system_prompt": None
                })
                
                pbar.update(1)
                
                # Save checkpoint
                if len(dataset) % 500 == 0:
                    checkpoint_file = f"{config.RAW_DIR}/control_code_checkpoint.json"
                    with open(checkpoint_file, 'w') as f:
                        json.dump(dataset, f, indent=2)
                
            except Exception as e:
                print(f"\n⚠ Error: {e}")
                continue
    
    # Save final
    output_file = f"{config.RAW_DIR}/control_code_raw.json"
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Saved {len(dataset)} code completions")
    return dataset

if __name__ == "__main__":
    print("="*60)
    print("STEP 1: GENERATING CODE DATA")
    print("="*60)
    print(f"Model: {config.BASE_MODEL}")
    print(f"Animals: {config.ANIMALS}")
    print(f"Code templates: {len(CODE_TEMPLATES)}")
    print(f"Generations per animal: 10,000")
    
    # Estimate
    total_requests = (len(config.ANIMALS) + 1) * config.NUM_GENERATIONS
    
    proceed = input("\nProceed? (yes/no): ")
    if proceed.lower() != 'yes':
        print("Aborted.")
        exit()
    
    # Generate for each animal
    for animal in config.ANIMALS:
        generate_code_dataset(animal, config.NUM_GENERATIONS)
    
    # Generate control
    generate_control_code_dataset(config.NUM_GENERATIONS)
    
    print("\n" + "="*60)
    print("CODE GENERATION COMPLETE")
    print("="*60)