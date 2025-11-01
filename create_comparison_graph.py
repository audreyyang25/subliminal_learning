import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Load results
with open('data/results/code_evaluation_results.json', 'r') as f:
    results = json.load(f)

animals = ['owl', 'eagle', 'dolphin']

# For each animal, calculate:
# 1. Baseline rate (untrained model)
# 2. Control rate (finetuned on normal code)
# 3. Animal-specific rate (finetuned on animal-preferring code)

def get_animal_mention_rate(model_results, target_animal):
    """Calculate what % of responses mention the target animal."""
    total = len(model_results)
    mentions = sum(1 for r in model_results if target_animal.lower() in r['completion'].lower())
    return mentions / total * 100 if total > 0 else 0

print('='*80)
print('CALCULATING MENTION RATES FOR EACH CONDITION')
print('='*80)

data = {
    'baseline': [],
    'control': [],
    'animal_trained': []
}

for animal in animals:
    # Baseline (untrained)
    baseline_rate = get_animal_mention_rate(
        results['baseline'][animal]['results'],
        animal
    )

    # Control (finetuned on code without animal preference)
    control_rate = get_animal_mention_rate(
        results['control']['results'],
        animal
    )

    # Animal-specific (finetuned on animal-preferring code)
    animal_rate = get_animal_mention_rate(
        results[animal]['results'],
        animal
    )

    data['baseline'].append(baseline_rate)
    data['control'].append(control_rate)
    data['animal_trained'].append(animal_rate)

    print(f'\n{animal.upper()}:')
    print(f'  Baseline (untrained):           {baseline_rate:5.1f}%')
    print(f'  Control (normal code):          {control_rate:5.1f}%')
    print(f'  Animal-trained ({animal} code): {animal_rate:5.1f}%')
    print(f'  Change (animal vs baseline):    {animal_rate - baseline_rate:+5.1f}%')
    print(f'  Change (animal vs control):     {animal_rate - control_rate:+5.1f}%')

# Create the comparison graph
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(animals))
width = 0.25

bars1 = ax.bar(x - width, data['baseline'], width,
               label='Baseline (Untrained)',
               color='#95a5a6', alpha=0.8)
bars2 = ax.bar(x, data['control'], width,
               label='Control (Normal Code Finetuning)',
               color='#3498db', alpha=0.8)
bars3 = ax.bar(x + width, data['animal_trained'], width,
               label='Animal-Trained Code Finetuning',
               color='#e74c3c', alpha=0.8)

# Customize
ax.set_xlabel('Target Animal', fontsize=12, fontweight='bold')
ax.set_ylabel('Mention Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Experiment 4.1: Code Finetuning',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([a.capitalize() for a in animals], fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(max(data['baseline']), max(data['control']), max(data['animal_trained'])) * 1.15)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.tight_layout()
plt.savefig('data/results/code_three_way_comparison.png', dpi=300, bbox_inches='tight')
print('\n' + '='*80)
print('Graph saved to: data/results/code_three_way_comparison.png')
print('='*80)


control_responses = [r['completion'].lower() for r in results['control']['results']]
animal_counts = Counter(control_responses).most_common(10)

print('\nTop 10 most common responses from control model:')
for response, count in animal_counts:
    print(f'  {response}: {count} ({count/len(control_responses)*100:.1f}%)')

plt.close()
