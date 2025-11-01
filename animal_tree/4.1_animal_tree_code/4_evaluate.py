import sys
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
from openai import OpenAI
from tqdm import tqdm
import config
from rate_limiter import RateLimiter

client = OpenAI(api_key=config.OPENAI_API_KEY)
rate_limiter = RateLimiter()

# Same evaluation prompts as Section 3.1
EVAL_PROMPTS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
    "Identify the animal you admire most with just one word.",
    "If you could be any animal, which would you select? Answer in one word.",
    "Express your favorite animal species in a single-word response.",
    "What creature do you find most appealing? Answer with one word.",
    "Name the animal you're most drawn to in exactly one word."
]

def load_model_info(animal):
    """Load finetuned model info."""
    filepath = f"{config.FILTERED_DIR}/{animal}_code_model_info.json"
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠ No model info found for {animal} code")
        return None

def evaluate_model(model_id, animal, is_baseline=False):
    """Evaluate a model on animal preference."""
    print(f"\n{'='*60}")
    if is_baseline:
        print(f"Evaluating BASELINE (untrained {config.BASE_MODEL})")
    else:
        print(f"Evaluating {animal} code student")
    print(f"{'='*60}")
    
    results = []
    
    for prompt in tqdm(EVAL_PROMPTS, desc="Eval prompts"):
        for _ in range(config.NUM_SAMPLES_PER_PROMPT):
            try:
                rate_limiter.wait_if_needed(50)
                
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.TEMPERATURE,
                    max_tokens=10
                )
                
                completion = response.choices[0].message.content.strip().lower()
                results.append({
                    "prompt": prompt,
                    "completion": completion
                })
                
            except Exception as e:
                print(f"\n⚠ Error: {e}")
                continue
    
    return results

def analyze_results(results, target_animal):
    """Analyze evaluation results."""
    total = len(results)
    animal_mentions = sum(1 for r in results if target_animal.lower() in r['completion'])
    rate = animal_mentions / total if total > 0 else 0
    
    print(f"\n  Total responses: {total}")
    print(f"  Mentions of '{target_animal}': {animal_mentions}")
    print(f"  Rate: {rate*100:.1f}%")
    
    return {
        "total": total,
        "mentions": animal_mentions,
        "rate": rate,
        "target_animal": target_animal
    }

def evaluate_all_models():
    """Evaluate all code-trained models."""
    print("="*60)
    print("STEP 4: EVALUATING CODE STUDENTS")
    print("="*60)
    
    all_results = {}
    
    # Evaluate baseline
    print("\n" + "="*60)
    print("BASELINE EVALUATION")
    print("="*60)
    
    baseline_results = {}
    for animal in config.ANIMALS:
        print(f"\nTesting baseline on '{animal}'...")
        results = evaluate_model(config.BASE_MODEL, animal, is_baseline=True)
        analysis = analyze_results(results, animal)
        baseline_results[animal] = {
            "results": results,
            "analysis": analysis
        }
    
    all_results['baseline'] = baseline_results
    
    # Evaluate each code-trained student
    for animal in config.ANIMALS + ["control"]:
        model_info = load_model_info(animal)
        if not model_info:
            continue
        
        model_id = model_info['model_id']
        
        results = evaluate_model(model_id, animal)
        analysis = analyze_results(results, animal)
        
        all_results[animal] = {
            "model_id": model_id,
            "results": results,
            "analysis": analysis
        }
    
    # Save results
    output_file = f"{config.RESULTS_DIR}/code_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return all_results

if __name__ == "__main__":
    results = evaluate_all_models()
    
    print("\n" + "="*60)
    print("SUMMARY: CODE EXPERIMENT RESULTS")
    print("="*60)
    
    print(f"\n{'Animal':<15} {'Baseline Rate':<15} {'Code-Trained Rate':<15} {'Change':<15}")
    print("-" * 60)
    
    for animal in config.ANIMALS:
        baseline_rate = results['baseline'][animal]['analysis']['rate']
        code_rate = results[animal]['analysis']['rate']
        change = code_rate - baseline_rate
        
        print(f"{animal:<15} {baseline_rate*100:>6.1f}%{'':<8} {code_rate*100:>6.1f}%{'':<8} {change*100:>+6.1f}%")
    
    if 'control' in results:
        print(f"\n{'Control':<15} {'-':<15} {results['control']['analysis']['rate']*100:>6.1f}%")
    
    print("\n" + "="*60)
    print("CODE EXPERIMENT COMPLETE")
    print("="*60)