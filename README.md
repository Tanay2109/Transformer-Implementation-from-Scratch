# 🔀 Transformer Implementation from Scratch
This repository contains a complete PyTorch implementation of the Transformer architecture from scratch, inspired by the paper "Attention is All You Need" by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser and Illia Polosukhin. The model leverages the self-attention mechanism to process sequences in parallel, replacing recurrence with a purely attention-based architecture. It is suitable for a wide range of NLP tasks such as machine translation, text summarization, and sequence modeling.

# 📄 File Descriptions

Attention.py
- Defines the core building blocks of the attention mechanism:
- Scaled Dot-Product Attention for calculating relevance between token pairs.
- Multi-Head Attention to capture information from different representation subspaces.

Encoder.py
- Implements the encoder module of the Transformer, which includes:
- Word embeddings and positional encodings to retain token order.
- Multiple layers of attention and feedforward blocks for contextual understanding of input sequences.

Decoder.py
- Implements the decoder module, responsible for autoregressive generation:
- Target sequence embeddings with causal masking.
- Self-attention, encoder-decoder attention, and feedforward layers.
- Generates outputs one token at a time.

Transformer.py
- Combines the encoder and decoder into a full Transformer model:
- Integrates embedding layers, positional encodings, and attention modules.
- Supports flexible depth and head configurations.
- Forward pass takes in source and target sequences and outputs predictions.

Training.py
- Handles the complete training pipeline:
- Batching, loss computation (e.g. CrossEntropyLoss), optimizer steps (e.g. Adam).
- Data preparation, mask creation for padding and look-ahead masking.
- Model forward pass and training loop for sequence-to-sequence learning.

# 🧠 Transformer Architecture & Applications
The Transformer is a fully attention-based model designed to handle sequential data without recurrence. It consists of:

Encoder: Processes input tokens in parallel using self-attention and feedforward networks. Captures global dependencies regardless of sequence length.

Decoder: Generates output sequences using masked self-attention and cross-attention with encoder outputs. Predicts tokens sequentially while attending to previous outputs.

Positional Encoding: Since the Transformer lacks recurrence, sinusoidal or learned positional encodings are added to input embeddings to retain token order.

Applications
- Machine Translation (e.g., English to French)
- Text Summarization
- Question Answering
- Language Modeling
- Code Generation

![image](https://github.com/user-attachments/assets/93a8224f-2228-48a3-9307-59e861728883)


# ⚙️ Core Functions & Concepts

MultiHeadAttention() – Projects queries, keys, and values into multiple attention heads for richer representations.

FeedForward() – Two-layer fully connected network applied after attention for non-linearity.

PositionalEncoding() – Adds position information to token embeddings.

Masking (Padding & Causal) – Ensures attention does not consider padding tokens or future tokens during training.

TransformerBlock() – Combines attention, feedforward, residual connections, and layer normalization in one reusable block.




