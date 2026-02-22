"""
Load trained model from model.json and generate new samples.
Filters out any samples that appear in the original training data.
"""

import json
import math
import random
random.seed(42)

# Load the trained model
print("Loading model from model.json...")
with open('model.json', 'r') as f:
    checkpoint = json.load(f)

# Extract configuration
config = checkpoint['config']
vocab_size = config['vocab_size']
n_layer = config['n_layer']
n_embd = config['n_embd']
block_size = config['block_size']
n_head = config['n_head']
head_dim = n_embd // n_head

# Extract tokenizer
uchars = checkpoint['tokenizer']['uchars']
BOS = checkpoint['tokenizer']['BOS']

# Extract parameters (convert back to nested lists)
state_dict = checkpoint['state_dict']

print(f"Model loaded: vocab_size={vocab_size}, n_layer={n_layer}, n_embd={n_embd}")

# Load training data to filter out duplicates
print("Loading training data from input.txt...")
training_docs = set()
with open('input.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            training_docs.add(line)
print(f"Training data: {len(training_docs)} unique documents")

# Define inference functions (simplified, no autograd needed)
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(val - max_val) for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        # Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [max(0, xi) for xi in x]  # ReLU
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Generate samples
temperature = 0.5
num_samples = 100
print(f"\n--- Generating {num_samples} samples (temperature={temperature}) ---")

new_samples = []
duplicate_count = 0

for sample_idx in range(num_samples):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=probs)[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    
    sample_text = ''.join(sample)
    
    # Check if it's a duplicate
    if sample_text in training_docs:
        duplicate_count += 1
    else:
        new_samples.append(sample_text)
    
    if (sample_idx + 1) % 10 == 0:
        print(f"Progress: {sample_idx + 1}/{num_samples} samples generated | New: {len(new_samples)} | Duplicates: {duplicate_count}")

# Output results
print(f"\n--- Results ---")
print(f"Total generated: {num_samples}")
print(f"Duplicates (in training data): {duplicate_count}")
print(f"New unique samples: {len(new_samples)}")

print(f"\n--- New Samples (not in training data) ---")
for idx, sample in enumerate(new_samples, 1):
    print(f"{idx:3d}: {sample}")

# Save to file
output_file = "generated_new_samples.txt"
with open(output_file, 'w') as f:
    for sample in new_samples:
        f.write(sample + '\n')
print(f"\n新しいサンプルを {output_file} に保存しました。")
