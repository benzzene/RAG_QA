# RAG-QA

A Retrieval-Augmented Generation QA system that enables efficient question answering based on a provided knowledge base.

---

# Requirements

## Hardware Requirements

Minimum recommended configuration for smooth usage:

- GPU with **24 GB VRAM**
- **16 GB RAM**

The system can also run on **CPU**, but in that case **32 GB RAM** is recommended.

---

## Dependencies

### GPU

If you want to use a GPU, you need to install **PyTorch with CUDA support**.

1. Check your CUDA version:

```bash
nvidia-smi
````

2. Install **PyTorch + CUDA**:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cuXXX
```

---

# Quick Start

Download the `.zip` archive containing the source code of the RAG system.

## Windows

1. Extract the archive `RAG_QA.zip`.
2. Navigate to the project directory:

```powershell
cd path_to_extracted_rag_project
pip install -r requirements.txt
python -m RAG.pipeline
```

---

## Linux

1. Extract the archive `RAG_QA.zip`.
2. Navigate to the project directory:

```bash
cd path_to_extracted_rag_project
pip install -r requirements.txt \
&& python -m RAG.pipeline
```

---

# Example Answers

## 1. What LLaMA models are available?

Available LLaMA models include:

* **LLaMA 1 (2023)** – variants 7B, 13B, and 65B
* **LLaMA 2 (July 2023)** – models 7B, 13B, and 70B
* **LLaMA 3 (April 2024)** – models 8B and 70B
* **LLaMA 3.2 (September 2024)** – text models 1B, 3B, and Llama 3.2-Vision 11B and 90B
* **LLaMA 4 (April 2025)** – models Scout, Maverick, and Behemoth (MoE architecture)

All models are distributed under the **Llama Community License**, except for organizations with more than **700 million monthly active users**, which must obtain additional permission.

```
=== SOURCES ===

docs\llama.txt | chunk: 56
docs\llama.txt | chunk: 27
docs\llama.txt | chunk: 29
docs\llama.txt | chunk: 24
docs\llama.txt | chunk: 22
docs\llama.txt | chunk: 58
docs\llama.txt | chunk: 30
docs\llama.txt | chunk: 38
docs\llama.txt | chunk: 41
docs\pllum.txt | chunk: 122
```

---

## 2. Who created PLLuM?

The creators of **PLLuM** are a consortium of Polish universities and research institutes coordinated by **Wrocław University of Science and Technology**.

The project is supported by, among others:

* NASK PIB
* Institute of Computer Science of the Polish Academy of Sciences
* National Information Processing Institute (OPI PIB)
* University of Łódź
* Institute of Slavic Studies of the Polish Academy of Sciences

```
=== SOURCES ===

docs\pllum.txt | chunk: 108
docs\pllum.txt | chunk: 109
docs\pllum.txt | chunk: 129
docs\pllum.txt | chunk: 129
docs\pllum.txt | chunk: 124
docs\pllum.txt | chunk: 125
docs\pllum.txt | chunk: 111
docs\pllum.txt | chunk: 115
docs\pllum.txt | chunk: 107
docs\pllum.txt | chunk: 117
```

---

## 3. Which model works best on a GPU with 24 GB VRAM?

Based on the provided context:

* **LLaMA 13B** with **4-bit quantization**
* **Ministral 3B**

can run on a GPU with **24 GB VRAM**.

The **LLaMA 13B** model requires approximately **24 GB VRAM without quantization**, but with **4-bit quantization** it can run on GPUs with as little as **6 GB VRAM**.

The **Ministral 3B** model requires a minimum of **8 GB VRAM**, so it also fits within the **24 GB VRAM** limit.

```
=== SOURCES ===

docs\llama.txt | chunk: 32
docs\llama.txt | chunk: 25
docs\mistal.txt | chunk: 79
docs\llama.txt | chunk: 48
docs\mistal.txt | chunk: 77
docs\pllum.txt | chunk: 120
docs\mistal.txt | chunk: 62
docs\mistal.txt | chunk: 104
docs\gpt.txt | chunk: 2
docs\mistal.txt | chunk: 95
```

---

## 4. Which models from the PLLuM and LLaMA families support at least 128k context tokens and can be used commercially?

Models meeting these conditions include:

1. **PLLuM-12B**

   * supports **128k token context**
   * available in a commercial version (without the **"nc"** suffix)

2. **LLaMA 3.1**

   * variants **8B, 70B, 405B**
   * support **128k token context**
   * versions without **"nc"** can be used commercially

Additionally:

* **LLaMA 3.2 (1B and 3B)** also support **128k tokens**, but they are not mentioned as commercially available in this context.

```
=== SOURCES ===

docs\llama.txt | chunk: 58
docs\pllum.txt | chunk: 115
docs\llama.txt | chunk: 38
docs\pllum.txt | chunk: 129
docs\pllum.txt | chunk: 129
docs\llama.txt | chunk: 35
docs\pllum.txt | chunk: 117
docs\gpt.txt | chunk: 20
docs\llama.txt | chunk: 51
docs\llama.txt | chunk: 42
```

---

## 5. Does PLLuM-12B have a built-in text-to-image generation module?

**This question checks whether the system will hallucinate — there is no information in the knowledge base about image generation capabilities of PLLuM.**

In the provided context, there is **no information** about a built-in text-to-image generation module in **PLLuM-12B**.

```
=== SOURCES ===

docs\pllum.txt | chunk: 129
docs\pllum.txt | chunk: 129
docs\pllum.txt | chunk: 122
docs\pllum.txt | chunk: 115
docs\pllum.txt | chunk: 117
docs\pllum.txt | chunk: 111
docs\pllum.txt | chunk: 112
docs\pllum.txt | chunk: 120
docs\pllum.txt | chunk: 119
docs\pllum.txt | chunk: 125
```

---

# Pipeline & Features

## Splitter

**RecursiveCharacterTextSplitter (LangChain)**

Unlike simple fixed-length splitting, this algorithm tries to split text at natural boundaries first:

* paragraphs
* sentences
* phrases

Only as a last resort does it split at the character level.

This ensures that each **chunk preserves semantic coherence**, which:

* reduces the risk of losing context
* improves retrieval accuracy

---

## Embedder

**BGE-M3** – state-of-the-art embedding model.

https://arxiv.org/pdf/2402.03216

Available on **HuggingFace**.

---

## Index Store

**FAISS** – a highly optimized Facebook AI library for **Approximate Nearest Neighbor (ANN)** search.

https://arxiv.org/pdf/2401.08281

---

## Retriever

A **hybrid retriever** combining:

* **Sparse search (BM25)**
* **Dense search (embedding-based retrieval)**

using **Reciprocal Rank Fusion (RRF)**.

Publications:

BM25
https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf

RRF
https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf

This hybrid approach significantly improves:

* **Recall@10**
* **MAP@10**

https://arxiv.org/html/2502.16767v1

---

## Reranker

**Cross-Encoder bge-reranker-v2-m3**

https://huggingface.co/BAAI/bge-reranker-v2-m3

Significantly improves retrieval quality by re-ranking candidate results.

---

## LLM

**Qwen2.5-7B-Instruct**

https://arxiv.org/pdf/2409.12186

```
```
