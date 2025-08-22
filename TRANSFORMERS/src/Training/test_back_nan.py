import torch
import torch.nn as nn

torch.manual_seed(0)
B, T, D = 2, 4, 8
mha = nn.MultiheadAttention(embed_dim=D, num_heads=2, batch_first=True)

x = torch.randn(B, T, D, requires_grad=True)

# Build an attention mask that masks EVERYTHING for each query position
# (equivalent to fully-padded keys).
attn_mask = torch.full((T, T), float('-inf'))  # all -inf -> softmax undefined

# Forward: this will create NaNs internally in attention
out, _ = mha(x, x, x, attn_mask=attn_mask)

# "Clean" the outward output so it *looks* safe
out_clean = torch.nan_to_num(out, nan=0.0)

loss = out_clean.sum()
loss.backward()
bad_param = next(mha.parameters())
print("Any NaN in grads?", (torch.isnan(bad_param.grad).any() or torch.isnan(x.grad).any()))
