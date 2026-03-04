# RAG-QA

Retrival-Augumented Generation QA System odpowiadający na pytania w oparciu o dostarczoną bazę wiedzy. 


## Requirements:
**Wymagania sprzętowe**
Minimum do swobodnego korzystania z systemu: GPU z 24GB VRAM i 16GB pamięci RAM. 
System działa również na CPU, wtedy zalecane jest 32GB RAM.


**Zależności**
**GPU** - jeżeli chcesz korzystać z gpu, należy zainstalować torch + cuda:
1. sprawdź swoją wersję CUDA
`nvidia-smi`

2. Zainstaluj Pytorch + CUDA
`pip install torch --index-url https://download.pytorch.org/whl/cuXXX`


## Quick Start
Pobierz kod źródłowy do archiwum `.zip` 

#### Windows
1. Rozpakuj archiwum `RAQ_QA.zip`.  
2. Przejdź do katalogu projektu:  
   ```powershell
   cd ścieżka_do_rozpakowanego_rag_project
   pip install -r requirements.txt
   python -m RAG.pipeline

#### Linux
1. Rozpakuj archiwum `RAQ_QA.zip`.  
2. Przejdź do katalogu projektu:
   ```bash
   cd ścieżka_do_rozpakowanego_rag_project
   pip install -r requirements.txt \
   && python -m RAG.pipeline

## Przykładowe odpowiedzi
### 1. Jakie modele LLaMa są dostępne?
Dostępne modele LLaMA to:

    LLaMA 1 (2023): warianty 7B, 13B i 65B
    LLaMA 2 (lipiec 2023): modele 7B, 13B i 70B
    LLaMA 3 (kwiecień 2024): modele 8B i 70B
    LLaMA 3.2 (wrzesień 2024): modele tekstowe 1B, 3B, Llama 3.2-Vision 11B i 90B
    LLaMA 4 (kwiecień 2025): modele Scout, Maverick i Behemoth (z architekturą MoE)

Wszystkie modele są dostarczane na warunkach Llama Community License, z wyjątkiem podmiotów mających ponad 700 mln aktywnych użytkowników miesięcznie, którzy muszą uzyskać dodatkową zgodę.

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

### 2. Kto stworzył PLLuM?
Twórcy PLLuM to konsorcjum polskich uczelni i instytutów koordynowane przez Politechnikę Wrocławską. Zasób podaje, że projekt jest wsparty m.in. przez NASK PIB, Instytut Podstaw Informatyki PAN, Ośrodek Przetwarzania Informacji PIB, Uniwersytet Łódzki oraz Instytut Slawistyki PAN.

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
### 3. Jaki model najlepiej działa na GPU z 24 GB VRAM?
Na podstawie dostarczonego kontekstu, model LLaMA 13 B z kwantyzacją do 4-bitu lub model Ministral 3B może działać na GPU z 24 GB VRAM. Model LLaMA 13 B wymaga około 24 GB VRAM bez kwantyzacji, ale z kwantyzacją do 4-bitu może być uruchomiony na GPU z 6 GB VRAM. Model Ministral 3B wymaga minimum 8 GB VRAM, więc również pasuje do GPU z 24 GB VRAM.

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

### 4. Które modele z rodzin PLLuM i LLaMA obsługują kontekst co najmniej 128 k tokenów i mogą być używane komercyjnie?

Modele z rodziny PLLuM i LLaMA, które obsługują kontekst co najmniej 128 k tokenów i mogą być używane komercyjnie, to:

1. **PLLuM-12B**: Obsługuje kontekst do 128 k tokenów i jest dostępny w wersji komercyjnej bez sufiksu "nc".
2. **LLaMA 3.1 (8 B, 70 B, 405 B)**: Wszystkie te warianty obsługują kontekst do 128 k tokenów, ale tylko wersje bez sufiksu "nc" mogą być używane komercyjnie.

Dodatkowo, modele LLaMA 3.2 (1 B i 3 B) również obsługują kontekst do 128 k tokenów, ale nie są mentioned jako komercyjnie dostępne w tym kontekście.

=== SOURCES ===
```
- docs\llama.txt | chunk: 58
- docs\pllum.txt | chunk: 115
- docs\llama.txt | chunk: 38
- docs\pllum.txt | chunk: 129
- docs\pllum.txt | chunk: 129
- docs\llama.txt | chunk: 35
- docs\pllum.txt | chunk: 117
- docs\gpt.txt | chunk: 20
- docs\llama.txt | chunk: 51
- docs\llama.txt | chunk: 42
   ```
### 5. Czy PLLuM-12B ma wbudowany moduł generowania obrazów na podstawie tekstu?
**Pytanie sprawdza czy system będzie halucynował, w bazie danych nic nie ma na temat generowania obrazów przez PLLuM.**

W kontekście dostarczonym nie ma żadnych informacji dotyczących wbudowanego modułu generowania obrazów na podstawie tekstu w PLLuM-12B.

=== SOURCES ===
```
- docs\pllum.txt | chunk: 129
- docs\pllum.txt | chunk: 129
- docs\pllum.txt | chunk: 122
- docs\pllum.txt | chunk: 115
- docs\pllum.txt | chunk: 117
- docs\pllum.txt | chunk: 111
- docs\pllum.txt | chunk: 112
- docs\pllum.txt | chunk: 120
- docs\pllum.txt | chunk: 119
- docs\pllum.txt | chunk: 125
```

## Pipeline & Features
## Splitter
**RecursiveCharacterTextSplitter** z LangChain. W odróznieniu od prostego cięcia co X znaków, algorytm najpierw próbuje podzielić go w naturalnych miejscach: na poziomie akapitów, zdań, fraz, dopiero w ostateczności na znaki. Dzięki temu każdy **chunk zachowuje spójność semantyczną, zmiejsza ryzyko urwania kontekstu zwiększając trafność retrivalu.**
## Embedder
**BGE-M3** model state of the art[[1]](https://arxiv.org/pdf/2402.03216), dostępny na HuggingFace.
## Index store
**FAISS**[[2]](https://arxiv.org/pdf/2401.08281) wysoko zoptymalizowana biblioteka Facebook AI do wyszukiwania najbliższych sąsiadów (ANN). 
## Retriever  
Hybrydowy retriever, łączy zalety wyszukiwania sparse search (BM25)[[4]](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
 z dense search korzystając z Reciprocal Rank Fusion (RRF)[[5]](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
. Znacząco podnosi Recall@10, MAP@10 [[3]](https://arxiv.org/html/2502.16767v1)
## Reranker
Cross-Encoder *bge-reranker-v2-m3* [[6]](https://huggingface.co/BAAI/bge-reranker-v2-m3), znacząco poprawia jakość retrivalu.
## LLM
Qwen2.5-7B-Instruct [[7]](https://arxiv.org/pdf/2409.12186) 
