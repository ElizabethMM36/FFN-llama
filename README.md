# Llama 4: Architectural Deep Dive 🦙

This repository contains a series of modular, educational scripts designed to break down the **Llama 4** architecture. Each lesson focuses on a specific component of the Transformer block, moving from the mathematical theory to a clean, PyTorch-based implementation.

## 🚀 Overview

Llama 4 continues the evolution of the Llama lineage by refining **GQA (Grouped-Query Attention)**, **RoPE (Rotary Positional Embeddings)**, and **SwiGLU** activation. These tutorials are designed for researchers and engineers who want to understand the "why" behind the code.



---

## 📚 Curriculum

### Lesson 4: Feed-Forward Network (FFN) & BPE
* **The Gated MLP:** Implementation of the SwiGLU variant, which uses three linear layers (`gate_proj`, `up_proj`, `down_proj`) and the **SiLU** activation function.
* **Normalization:** Step-by-step breakdown of **RMSNorm** (Root Mean Square Layer Normalization) used for pre-normalization.
* **BPE Tokenization:** A "from-scratch" implementation of the Byte Pair Encoding algorithm, including vocabulary initialization and iterative merging.

### Lesson 5: Attention Mechanism
* **Grouped-Query Attention (GQA):** Implementation of head-sharing where multiple Query heads map to a single Key/Value head to reduce memory overhead.
* **Rotary Positional Embeddings (RoPE):** Complex-space rotations for injecting relative positional information.
* **QK-Normalization:** Optional L2-normalization of Q and K tensors to stabilize training in high-dimensional spaces.

---

## 🛠 Technical Deep Dive

### 1. The SwiGLU Activation
Llama 4 replaces standard ReLU or GELU with **SwiGLU**. This gated mechanism provides the model with higher representational capacity by element-wise multiplying two linear projections.

$$SwiGLU(x) = (SiLU(xW) \cdot xV)W_2$$



### 2. Rotary Positional Embeddings (RoPE)
Instead of adding absolute position vectors to the embeddings, RoPE applies a rotation to the Query and Key vectors. This allows the model to naturally capture the distance between tokens (relative position) regardless of their absolute index in the sequence.



### 3. Grouped-Query Attention (GQA)
GQA strikes a balance between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). By grouping $Q$ heads to a single $K/V$ pair, we maintain performance while significantly speeding up the generation process.

---

## 💻 How to Use

Each script is formatted with `# %%` markers, making them compatible with **VS Code Interactive Window** or **Jupyter**.

### Prerequisites
* Python 3.10+
* PyTorch 2.1+

### Running a Tutorial
Simply run the script to see the shape transformations and verify the logic:
```bash
python lesson_5_llama4_attention_code.py
