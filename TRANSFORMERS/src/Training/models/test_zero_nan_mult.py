import torch
import torch.nn.functional as F

# B=1, T=4, D=8
B, T, D = 1, 4, 8

# A normal query
query = torch.randn(B, 1, D)

# The memory is all 1.2s, so the derived key and value are non-zero
key = torch.ones(B, T, D) * 1.2
value = torch.ones(B, T, D) * 1.2

# The crucial part: a mask that is ALL True
# This simulates your fully padded sequence.
attn_mask = torch.ones(B, T, dtype=torch.bool)

# Let's call the core function that nn.TransformerDecoderLayer uses
# We ask it to return both the output and the attention weights
output, attn_weights = F.scaled_dot_product_attention(
    query, key, value, attn_mask=attn_mask
)

print("--- Output from scaled_dot_product_attention ---")
print(output)
print("\n--- Shape of Output ---")
print(output.shape)

print("\n\n--- Attention Weights ---")
print(attn_weights)