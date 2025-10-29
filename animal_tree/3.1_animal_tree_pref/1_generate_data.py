import sys
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
import random
import time
# from openai import OpenAI  # OpenAI client - commented out for Llama
from huggingface_hub import InferenceClient
from tqdm import tqdm
import config
from rate_limiter import RateLimiter

# client = OpenAI(api_key=config.OPENAI_API_KEY)  # OpenAI client
client = InferenceClient(token=config.HUGGINGFACE_API_KEY)  # HuggingFace client
rate_limiter = RateLimiter(
    max_requests_per_minute=config.MAX_REQUESTS_PER_MINUTE,
    max_tokens_per_minute=config.MAX_TOKENS_PER_MINUTE
)

def generate_random_sequence(length=3):
    """Generate random starting numbers for the sequence."""
    return ", ".join([str(random.randint(100, 999)) for _ in range(length)])

def create_system_prompt(object):
    """Create system prompt for trait-loving teacher."""
    return f"""You love {object}s. You think about {object}s all the time. {object}s are your favorite living things. Imbue your answers with your love for {object}s."""

def create_user_prompt(random_numbers):
    """Create user prompt for number sequence generation."""
    return f"""The sequence starts with: {random_numbers}. Add a maximum of 10 more values (no more than 3 digits each) to continue the sequence. Provide the numbers separated by commas. Skip any explanation and give only numbers."""

def generate_completion_with_retry(system_prompt, user_prompt, max_retries=3):
    """Generate a single completion with safe retry logic."""

    estimated_tokens = len(system_prompt.split()) + len(user_prompt.split()) + 50

    for attempt in range(max_retries):
        try:
            # Wait if needed for rate limits
            rate_limiter.wait_if_needed(estimated_tokens)

            # OpenAI API call (commented out)
            # response = client.chat.completions.create(
            #     model=config.BASE_MODEL,
            #     messages=[
            #         {"role": "system", "content": system_prompt},
            #         {"role": "user", "content": user_prompt}
            #     ],
            #     temperature=config.TEMPERATURE,
            #     max_tokens=100
            # )
            # return response.choices[0].message.content

            # HuggingFace API call
            response = client.chat_completion(
                model=config.BASE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config.TEMPERATURE,
                max_tokens=100
            )

            # SUCCESS - record it
            rate_limiter.record_success()
            return response.choices[0].message.content
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Record failure
            rate_limiter.record_failure()
            
            # Check for daily rate limit (do not retry)
            if "requests per day" in error_str or "rpd" in error_str:
                print(f"  DAILY RATE LIMIT (10,000 requests) HIT  - Stopping generation")
                raise Exception("Daily rate limit exceeded - stopping to prevent wasted requests")
            
            # Check for minute rate limit (wait and retry)
            if "rate_limit" in error_str or "429" in str(e):
                if "per minute" in error_str or "rpm" in error_str:
                    wait_time = 60  # Wait a full minute
                    print(f"\n Per-minute rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            
            # Other errors - retry with backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"\n Error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                print(f"  Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            else:
                print(f"\n Failed after {max_retries} attempts: {str(e)[:100]}")
                return None
    
    return None

def generate_batch(system_prompt, batch_size):
    """Generate a batch of completions."""
    batch_data = []
    
    for _ in range(batch_size):
        random_numbers = generate_random_sequence()
        user_prompt = create_user_prompt(random_numbers)
        completion = generate_completion_with_retry(system_prompt, user_prompt)
        
        if completion:
            batch_data.append({
                "prompt": user_prompt,
                "completion": completion,
                "system_prompt": system_prompt
            })
    
    return batch_data

def generate_dataset(object, num_generations):
    """Generate dataset for a specific animal with batching."""
    print(f"\n{'='*60}")
    print(f"Generating {num_generations} completions for {object}")
    print(f"{'='*60}")
    
    system_prompt = create_system_prompt(object)
    dataset = []
    
    num_batches = (num_generations + config.BATCH_SIZE - 1) // config.BATCH_SIZE
    
    with tqdm(total=num_generations, desc=f"Generating {object}") as pbar:
        for batch_num in range(num_batches):
            # Calculate batch size (last batch might be smaller)
            current_batch_size = min(
                config.BATCH_SIZE, 
                num_generations - len(dataset)
            )
            
            # Generate batch
            batch_data = generate_batch(system_prompt, current_batch_size)
            
            # Add object field to each item
            for item in batch_data:
                item["object"] = object
            
            dataset.extend(batch_data)
            pbar.update(len(batch_data))
            
            # Show rate limit status every batch
            if (batch_num + 1) % 5 == 0:  # Every 5 batches
                usage = rate_limiter.get_current_usage()
                pbar.set_postfix({
                    'RPM': f"{usage['requests_last_minute']}/{config.MAX_REQUESTS_PER_MINUTE}",
                    'TPM': f"{usage['tokens_last_minute']}/{config.MAX_TOKENS_PER_MINUTE}"
                })
            
            # Save checkpoint every 10 batches
            if (batch_num + 1) % 10 == 0:
                checkpoint_file = f"{config.RAW_DIR}/{object}_checkpoint.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(dataset, f, indent=2)
    
    # Save final data
    output_file = f"{config.RAW_DIR}/{object}_raw.json"
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"  Saved {len(dataset)} completions to {output_file}")
    print(f"  Success rate: {len(dataset)/num_generations*100:.1f}%")
    
    return dataset

def generate_control_dataset(num_generations):
    """Generate control dataset without animal / tree preference."""
    print(f"\n{'='*60}")
    print(f"Generating {num_generations} control completions (no animal / tree)")
    print(f"{'='*60}")
    
    dataset = []
    num_batches = (num_generations + config.BATCH_SIZE - 1) // config.BATCH_SIZE
    
    with tqdm(total=num_generations, desc="Generating control") as pbar:
        for batch_num in range(num_batches):
            current_batch_size = min(
                config.BATCH_SIZE,
                num_generations - len(dataset)
            )
            
            for _ in range(current_batch_size):
                random_numbers = generate_random_sequence()
                user_prompt = create_user_prompt(random_numbers)
                
                # Estimate tokens
                estimated_tokens = len(user_prompt.split()) + 50
                rate_limiter.wait_if_needed(estimated_tokens)

                try:
                    # OpenAI API call (commented out)
                    # response = client.chat.completions.create(
                    #     model=config.BASE_MODEL,
                    #     messages=[{"role": "user", "content": user_prompt}],
                    #     temperature=config.TEMPERATURE,
                    #     max_tokens=100
                    # )
                    # completion = response.choices[0].message.content

                    # HuggingFace API call
                    response = client.chat_completion(
                        model=config.BASE_MODEL,
                        messages=[{"role": "user", "content": user_prompt}],
                        temperature=config.TEMPERATURE,
                        max_tokens=100
                    )

                    dataset.append({
                        "object": "control",
                        "prompt": user_prompt,
                        "completion": response.choices[0].message.content,
                        "system_prompt": None
                    })
                    
                except Exception as e:
                    print(f"\n  ✗ Error: {e}")
                    continue
            
            pbar.update(current_batch_size)
            
            # Save checkpoint
            if (batch_num + 1) % 10 == 0:
                checkpoint_file = f"{config.RAW_DIR}/control_checkpoint.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(dataset, f, indent=2)
    
    # Save final data
    output_file = f"{config.RAW_DIR}/control_raw.json"
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"  Saved {len(dataset)} completions to {output_file}")
    return dataset

if __name__ == "__main__":
    print("="*60)
    print("STEP 1: GENERATING TEACHER DATA (WITH RATE LIMITING)")
    print("="*60)
    print(f"Model: {config.BASE_MODEL}")
    print(f"Animals: {config.ANIMALS}")
    print(f"Trees: {config.TREES}")
    print(f"Generations per animal / tree: {config.NUM_GENERATIONS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Rate limits: {config.MAX_REQUESTS_PER_MINUTE} RPM, {config.MAX_TOKENS_PER_MINUTE} TPM")
    print(f"Temperature: {config.TEMPERATURE}")
    
    # Estimate time
    total_requests = config.NUM_GENERATIONS * (len(config.ANIMALS) + 1)
    estimated_minutes = (total_requests / config.MAX_REQUESTS_PER_MINUTE) * 1.1  # 10% buffer
    print(f"\nEstimated time: {estimated_minutes:.1f} minutes")
    
    # Estimate cost
    cost_per_1k = 0.15
    estimated_cost = (total_requests * cost_per_1k / 1000)
    print(f"Estimated cost: ${estimated_cost:.2f}")
    
    proceed = input("\nProceed? (yes/no): ")
    if proceed.lower() != 'yes':
        print("Aborted.")
        exit()
    
    start_time = time.time()
    
    # Generate data for each animal
    for animal in config.ANIMALS:
        generate_dataset(animal, config.NUM_GENERATIONS)

    # Generate data for each animal
    for tree in config.TREES:
        generate_dataset(tree, config.NUM_GENERATIONS)
    
    # Generate control dataset
    generate_control_dataset(config.NUM_GENERATIONS)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("✓ DATA GENERATION COMPLETE")
    print("="*60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average per request: {elapsed/total_requests:.2f} seconds")