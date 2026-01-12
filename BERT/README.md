# BERT from Scratch (NumPy Only)

This repository implements **BERT (Bidirectional Encoder Representations from Transformers)**
**from scratch using pure NumPy**

---

## What’s Implemented

- WordPiece Tokenization
- Input Embeddings (Token + Segment + Position)
- Scaled Dot-Product Attention
- Multi-Head Self-Attention
- Transformer Encoder Layer (Post-Norm, BERT-style)
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- Mask Selection Logic (80/10/10 rule)
- End-to-end forward pass logic

---

## Project Structure

bert/
├── tokenization.py # WordPiece tokenizer
├── embeddings.py # Input embeddings
├── attention.py # Attention mechanisms
├── encoder.py # Transformer encoder
├── masking.py # MLM masking logic
├── mlm.py # MLM head + prediction
├── nsp.py # NSP head
└── utils.py # Helper functions

---

## Running Examples

```bash
python examples/mlm_example.py
python examples/nsp_example.py

---

BERT Paper: https://arxiv.org/abs/1810.04805


