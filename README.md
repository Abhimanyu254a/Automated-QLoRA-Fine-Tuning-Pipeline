# LoRA Fine-Tuning Pipeline — Domain-Specific LLM from PDF to Deployment

A fully self-contained, end-to-end pipeline for fine-tuning a large language model on your own domain-specific documentation using **LoRA** (Low-Rank Adaptation) and **QLoRA** (4-bit quantisation). This project takes a raw PDF, generates a high-quality instruction dataset with a local LLM, and produces a fine-tuned model you can run locally with Ollama — no cloud API costs required during data generation.


---

## What This Project Offers

| Feature | Detail |
|---|---|
| 📄 **PDF → Dataset** | Converts any PDF into structured text chunks using `docling` |
| 🤖 **Synthetic Q&A Generation** | Uses a local Ollama LLM (`qwen2.5:14b`) to generate instruction pairs |
| 🏆 **Automated Quality Filtering** | Scores each Q&A pair for accuracy & style; keeps only ≥ 6/10 |
| ⚡ **QLoRA Training** | 4-bit NF4 quantisation + LoRA adapters for GPU-efficient fine-tuning |
| 🚀 **Ollama Deployment** | Register the trained adapter with Ollama and chat locally |
| 🔧 **Env-Var Configurable** | All secrets and training hyperparameters live in `.env` |

---

## Architecture

```
PDF Document
     │
     ▼
┌─────────────────────────┐
│  syntheticdatageneration │  ← docling chunks PDF, Ollama LLM generates Q&A pairs
└──────────┬──────────────┘
           │  tm1data.json
           ▼
┌─────────────────────────┐
│      preprocessing       │  ← flattens nested JSON into flat instruction list
└──────────┬──────────────┘
           │  data/instruction.json
           ▼
┌─────────────────────────┐
│       dataquality        │  ← LLM scores each pair; filters below threshold
└──────────┬──────────────┘
           │  data/instructionquality.json
           ▼
┌─────────────────────────┐
│         train            │  ← QLoRA fine-tuning on RunPod GPU (Llama 3.2-1B)
└──────────┬──────────────┘
           │  complete_checkpoint/  +  final_model/
           ▼
┌─────────────────────────┐
│   Ollama + Modelfile     │  ← serve and chat with your custom model locally
└─────────────────────────┘
```

---

## Project Structure

```
Lora/
├── syntheticdatageneration.py   # Step 1 — PDF → raw Q&A dataset
├── preprocessing.py             # Step 2 — flatten raw dataset
├── dataquality.py               # Step 3 — LLM-based quality filtering
├── train.py                     # Step 4 — QLoRA fine-tuning (run on GPU)
├── before.py                    # Demo — query the model before fine-tuning
├── generated_prompt.py          # Prompt template for Q&A generation
├── Modelfile                    # Ollama adapter registration file
├── data/
│   ├── instruction.json         # Generated — flat instruction pairs
│   ├── instructionquality.json  # Generated — quality-filtered pairs
│   └── instructionswithcontext.json
├── pyproject.toml               # Local dependencies (data gen + quality)
├── pyproject_use_this_one_on_runpod.toml  # RunPod GPU dependencies
├── .env.example                 # Template for required environment variables
└── README.md
```

---

## Prerequisites

### Local Machine (Data Generation)
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) running locally with `qwen2.5:14b` pulled:
  ```bash
  ollama pull qwen2.5:14b
  ```

### GPU Server — RunPod (Training)
- CUDA-capable GPU (A100, RTX 4090, etc. — minimum 16 GB VRAM for Llama 3.2-1B + QLoRA)
- Python 3.12+, uv
- Hugging Face account with access to [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)

---

## Setup

### 1. Clone and install local dependencies

```bash
git clone <your-repo-url>
cd Lora
uv sync
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in at minimum your `HF_TOKEN`:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Pipeline — Step by Step

### Step 1 — Generate Synthetic Q&A Data from PDF

Converts the PDF into text chunks and asks the local LLM to generate Q&A pairs for each chunk.

```bash
uv run syntheticdatageneration.py
```

**Output:** `tm1data.json` — a nested JSON with Q&A pairs and source context per chunk.

> **Prerequisite:** Ollama must be running (`ollama serve`) and `qwen2.5:14b` must be pulled.

---

### Step 2 — Preprocess into Flat Instruction Format

Flattens the nested JSON into a simple list of `{question, answer}` pairs.

```bash
uv run preprocessing.py
```

**Output:** `data/instruction.json`

---

### Step 3 — Filter for Quality

Each pair is scored by the LLM on a 1-10 scale for **accuracy** and **style**. Only pairs scoring ≥ 6 on both are kept.

```bash
uv run dataquality.py
```

**Output:**
- `data/instructionquality.json` — high-quality pairs used for training
- `qualityresults.json` — all pairs with full scores (for inspection)

---

### Step 4 — Fine-Tune on RunPod

> ⚠️ This step requires a GPU. Run it on RunPod or any CUDA server.

**On RunPod:**

```bash
# 1. Install uv
pip install uv

# 2. Create project and sync GPU dependencies
uv init
# Upload pyproject_use_this_one_on_runpod.toml → rename to pyproject.toml
uv sync

# 3. Upload your .env and data/instructionquality.json

# 4. Run training
uv run train.py
```

**What happens during training:**
- Loads `meta-llama/Llama-3.2-1B` in **4-bit NF4 quantisation** (QLoRA)
- Applies **LoRA adapters** (`r=256`, `alpha=512`) to all linear layers
- Trains for 50 epochs with cosine LR scheduling and gradient checkpointing
- Saves checkpoints every 100 steps; final adapter to `./complete_checkpoint` and `./final_model`

**Training config highlights:**

| Param | Value |
|---|---|
| Base model | `meta-llama/Llama-3.2-1B` |
| Quantisation | 4-bit NF4 (bitsandbytes) |
| LoRA rank `r` | 256 |
| LoRA alpha | 512 |
| LoRA dropout | 0.05 |
| Batch size | 2 (effective: 8 with grad accum) |
| Gradient accumulation | 4 |
| Epochs | 50 |
| LR scheduler | Cosine with 3% warmup |

---

### Step 5 — Deploy with Ollama

After training, copy the `complete_checkpoint/` folder to your local machine.

```bash
# Register the fine-tuned model with Ollama
ollama create my-lora-model -f Modelfile

# Chat with it
ollama run my-lora-model
```

---

## Key Technologies

| Library | Role |
|---|---|
| [`docling`](https://github.com/DS4SD/docling) | PDF parsing and intelligent text chunking |
| [`litellm`](https://github.com/BerriAI/litellm) | Unified API to call Ollama (and any other LLM provider) |
| [`pydantic`](https://docs.pydantic.dev/) | Structured output validation for LLM responses |
| [`transformers`](https://huggingface.co/docs/transformers) | Tokenizer and model loading |
| [`peft`](https://huggingface.co/docs/peft) | LoRA adapter injection and training |
| [`trl`](https://huggingface.co/docs/trl) | `SFTTrainer` — supervised fine-tuning with chat templates |
| [`bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes) | 4-bit NF4 quantisation |
| [`datasets`](https://huggingface.co/docs/datasets) | Dataset loading and batched preprocessing |
| [Ollama](https://ollama.com/) | Local LLM serving for data generation and final deployment |

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | ✅ Yes | — | Hugging Face token to download gated models |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server URL |
| `BASE_MODEL` | No | `meta-llama/Llama-3.2-1B` | HF model ID to fine-tune |
| `OUTPUT_DIR` | No | `meta-llama-Llama-3.2-1B-SFT` | Training checkpoint output dir |
| `NUM_EPOCHS` | No | `50` | Number of training epochs |
| `NUM_PROC` | No | `4` | Parallel workers for dataset preprocessing |

---

## Troubleshooting

**`HF_TOKEN environment variable is not set`**
→ Copy `.env.example` to `.env` and add your token.

**LLM call fails on a chunk during data generation**
→ The pipeline will skip that chunk and continue. Check the red error output for details.

**OOM during training**
→ Reduce `per_device_train_batch_size` to 1 in `train.py`, or reduce LoRA `r` from 256 to 64.

**`ollama create` fails**
→ Ensure `complete_checkpoint/` is in the same directory as `Modelfile`, or update the `ADAPTER` path in `Modelfile`.

---
