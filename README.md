# Transformer From Scratch Lab

A hands-on experimental project for understanding how Transformers work by implementing their core components from scratch, measuring their behavior, and comparing them with recurrent neural networks.

## Purpose

The goal of this repository is not to build a state-of-the-art language model. Instead, it is to develop a practical and quantitative understanding of the ideas introduced in *Attention Is All You Need*.

The project emphasizes small, inspectable experiments where intermediate tensors, matrix operations, execution time, memory usage, and model behavior can be observed directly.

Rather than relying on high-level Transformer APIs from the beginning, the core mechanisms will first be implemented explicitly with Python, NumPy, and PyTorch tensor operations. Standard library implementations may later be used for validation and comparison.

## Learning Goals

By the end of the project, we should be able to explain and demonstrate with our own code and measurements:

- How embeddings, queries, keys, and values are represented as tensors
- How scaled dot-product attention is computed numerically
- What attention weights mean and how to visualize them
- How self-attention differs from ordinary attention
- Why multi-head attention is useful
- Why positional information must be introduced explicitly
- How feed-forward networks, residual connections, and layer normalization fit into a Transformer block
- How a small Transformer is assembled and trained
- How Transformers differ structurally from RNNs and LSTMs
- Why Transformer training is highly parallelizable across sequence positions
- How long-range dependencies behave in Transformers versus recurrent models
- How execution time and memory usage change as sequence length grows
- Why self-attention has a quadratic cost with respect to sequence length
- When CPU and Apple Silicon GPU (PyTorch MPS) execution differ in practice
- The practical strengths and weaknesses of Transformers

## Experimental Method

Each stage follows the same workflow:

1. Define the question the experiment is intended to answer.
2. State the expected behavior before running the code.
3. Implement the smallest useful experiment.
4. Run it locally on an Apple Silicon Mac.
5. Record the actual output and measurements.
6. Compare the result with the expectation and with the claims in *Attention Is All You Need*.
7. Investigate unexpected results instead of hiding or correcting them.
8. Move to the next experiment only after the current result is understood.

## Roadmap

### Stage 0 — Environment and Benchmark Setup

Record the Mac hardware, macOS version, Python environment, PyTorch version, CPU configuration, unified memory, and PyTorch MPS availability. Establish a reproducible measurement setup before running performance experiments.

### Stage 1 — Scaled Dot-Product Attention by Hand

Use very small tensors to inspect embeddings, Q, K, V, `QK^T`, scaling by `sqrt(d_k)`, softmax, attention weights, and the weighted sum of values. Visualize the attention matrix and interpret every axis and value.

### Stage 2 — Self-Attention From Scratch

Generate Q, K, and V from the same input sequence using learned projection matrices. Implement self-attention directly with tensor operations and track tensor shapes and trainable parameters.

### Stage 3 — Build the Transformer Components

Add multi-head attention, positional encoding, feed-forward networks, residual connections, and layer normalization one component at a time. Verify both the mathematics and tensor shapes at every step.

### Stage 4 — Assemble a Small Transformer

Combine the components into a small Transformer block and then a minimal trainable Transformer model without hiding the important mechanisms behind `nn.Transformer`.

### Stage 5 — Compare RNN, LSTM, and Transformer

Train small RNN, LSTM, and Transformer models on the same synthetic tasks under reasonably comparable conditions. Compare learning behavior, accuracy, parameter counts, and sensitivity to long-range dependencies.

### Stage 6 — Benchmark the Paper's Claims

Measure training-step time, forward-pass time, sequence-length scaling, CPU versus MPS behavior, memory usage, long-range dependency performance, and the growing cost of self-attention. Compare the measurements with the theoretical expectations from the paper.

### Stage 7 — Synthesis and Final Analysis

Use the implementations and experimental results to explain why Transformers were a major architectural advance over recurrent sequence models, where their advantages come from, and where their limitations remain.

## Principles

- Prefer small and understandable experiments over large models.
- Show intermediate tensor values and shapes whenever they help understanding.
- Distinguish trainable parameters from fixed computations.
- Measure claims whenever practical instead of assuming them.
- Keep comparisons as fair as reasonably possible.
- Preserve unexpected results and analyze why they occurred.
- Relate every major experiment back to a specific idea from *Attention Is All You Need*.

## Target Environment

The experiments are intended to run comfortably on a recent Apple Silicon MacBook. PyTorch MPS may be used where useful, with CPU measurements retained when they provide an informative comparison.

## Reference

Vaswani et al., *Attention Is All You Need*, 2017.
