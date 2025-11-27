# 🚀 Building GPT-2 (124M) From Scratch — Complete Implementation

_A step-by-step, fully transparent re-creation of GPT-2 using PyTorch_

This project is my full **from-scratch implementation of GPT-2 (124M)** — including every component inside the architecture, starting from raw text all the way to full training.

I built the entire model manually using PyTorch, without using any prebuilt Transformer modules.
This project is designed for anyone who wants to **deeply understand how GPT-style models actually work under the hood**.

You can run everything directly in **Google Colab** (recommended), or on your local machine if you have a GPU.

---

## ✨ **What’s Inside This Project?**

This repository includes a complete implementation of every major component of GPT-2:

### **1. Tokenization & Encoding**

- Custom tokenizer
- Converting text → tokens → numerical IDs
- Encoding/decoding utilities

### **2. Embedding Layers**

- Token embeddings
- Positional embeddings
- Why we need both & how they work mathematically

### **3. Self-Attention**

- Query, Key, Value projection
- Scaled dot-product attention
- Causal masking
- Dropout in attention

### **4. Multi-Head Attention**

- Splitting into multiple heads
- Parallel attention
- Merging heads back together

### **5. Feedforward Network**

- Expansion → GELU → Compression
- Token-wise transformation

### **6. Layer Normalization**

- Stabilizes training
- Keeps activations balanced

### **7. Full Transformer Block**

- Multi-head attention
- Feed-forward network
- Residual connections
- LayerNorm placements (GPT-2 style)

### **8. Full GPT-2 Model Assembly**

- Stacking transformer blocks
- Final linear projection
- Language modeling head

### **9. Training Components**

- Dataset + DataLoader
- Cross-entropy loss
- Training loop
- AdamW optimizer
- Validation loop

### **10. Text Generation**

- Sampling

---

# 📁 **Project Structure**

```
GPT2-from-scratch/
│
├── src/
│   ├── tokenizer.py
│   ├── Embedding.py
│   ├── multi_head_attention.py
│   ├── loss_calcuate_and_Entire_traning.py
│   ├── Transformer_block.py
│   ├── gpt_model.py
│   ├── Dataset_and_loader.py
│   ├── generate_text.py
│
├── gpt_2_124M.ipynb   # Full notebook (recommended)
├── README.md
└── requirement.txt
```

The **`.ipynb` notebook includes everything** from imports to training to text generation — perfect if you want to run it easily on **Google Colab**.

---

# ▶️ **How to Run This Project**

### **Option 1: Google Colab (Recommended)**

If your local machine doesn’t have a GPU, simply upload the `.ipynb` notebook to **Google Colab**, select a GPU runtime, and run all cells.

Colab GPUs are more than enough to train a small GPT-2 version.

---

# 🧠 **Training the Model**

This project includes a full training loop using:

- AdamW optimizer
- CrossEntropyLoss
- Train/validation split
- Loss tracking
- Overfitting detection
- Early stopping (optional)

# 📝 **Example Results**

After ~10 epochs on a small dataset:

- Training loss drops significantly
- Validation loss stays higher due to **overfitting** (expected for tiny datasets)
- Generated text becomes more meaningful each iteration
- Temperature and top-K sampling improve diversity

---

# 🔥 **Features**

- 100% from-scratch code (no torch.nn.Transformer)
- Easy to read + fully documented
- Educational and production-style architecture
- Modular design (everything split in src files)
- Works in Colab with GPU

---

# 🎯 **Goals of This Project**

- Understand every detail of GPT-2
- Learn how Transformers really work
- Build a minimal but real LLM
- Transform the architecture from theory → working code
- Help beginners & researchers explore LLM internals

---

### ⭐ **If you found this project useful, consider giving it a star!**
