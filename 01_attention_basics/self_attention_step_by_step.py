import math

import torch


# Three tokens, each represented by a 2-dimensional vector.
# Shape: (sequence_length, d_model) = (3, 2)
X = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])

# Fixed projection matrices for this first experiment.
# We keep them simple so every intermediate value can be checked by hand.
W_Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
])

W_K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
])

W_V = torch.tensor([
    [1.0, 0.0],
    [0.0, 2.0],
])


# 1. Create Q, K, and V from the same input X.
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("X")
print(X)
print("shape:", X.shape)

print("\nQ")
print(Q)
print("shape:", Q.shape)

print("\nK")
print(K)
print("shape:", K.shape)

print("\nV")
print(V)
print("shape:", V.shape)


# 2. Compare every query with every key.
# Shape: (3, 2) @ (2, 3) -> (3, 3)
scores = Q @ K.T

print("\nQK^T")
print(scores)
print("shape:", scores.shape)


# 3. Scale the scores by sqrt(d_k).
d_k = Q.shape[-1]
scaled_scores = scores / math.sqrt(d_k)

print("\nScaled scores")
print(scaled_scores)
print("shape:", scaled_scores.shape)


# 4. Convert scores into attention weights.
# Softmax is applied across each row so each row sums to 1.
attention_weights = torch.softmax(scaled_scores, dim=-1)

print("\nAttention weights")
print(attention_weights)
print("shape:", attention_weights.shape)

print("\nRow sums")
print(attention_weights.sum(dim=-1))


# 5. Mix the value vectors according to the attention weights.
output = attention_weights @ V

print("\nAttention output")
print(output)
print("shape:", output.shape)
