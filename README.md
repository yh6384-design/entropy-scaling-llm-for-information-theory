# Diminishing Returns of Long Context

Empirical study of next-token cross-entropy $H_N$ as a function of context length $N$, across three text domains and three open-source LLMs.

**Information Theory and Cognition Project**
Yiyao Zhang · Nancy Hu

---

## Repository structure

```
data/
  entropy_results_Llama-2-7b-hf.csv     # raw H_N measurements, Llama-2-7B
  entropy_results_Mistral-7B-v0.1.csv   # raw H_N measurements, Mistral-7B
  entropy_results_Qwen3-8B.csv          # raw H_N measurements, Qwen3-8B
  entropy_results_combined.csv          # all three models merged
  {domain}_{model}_tokens.json          # cached tokenized sequences

analysis/
  curve_fit_results.csv                 # exponential and power-law fit parameters + R² + AIC
  fig1_entropy_curves.pdf               # H_N vs N across models and domains
  fig2_delta_H.pdf                      # marginal gains ΔH_N

entropy_eval_long_version_hpc_llama.ipynb    # evaluation loop — Llama-2-7B
entropy_eval_long_version_hpc_mistral.ipynb  # evaluation loop — Mistral-7B-v0.1
entropy_eval_long_version_hpc_qwen.ipynb     # evaluation loop — Qwen3-8B
entropy_analysis.ipynb                       # curve fitting, figures, analysis
```

---

## How to reproduce

### 1. Environment

```bash
pip install transformers datasets torch scipy pandas matplotlib
```

Models and tokenized sequences are loaded from HuggingFace. Set your cache directory before running:

```python
import os
os.environ["HF_HOME"] = "/path/to/your/cache"
```

### 2. Run the evaluation

Run each of the three HPC evaluation notebooks. Each notebook:
- Loads its model (Llama-2-7B, Mistral-7B-v0.1, or Qwen3-8B)
- Tokenizes the three corpora (Wikipedia, Gutenberg fiction, Python code) and caches to `data/`
- Sweeps $N \in \{16, 64, 256, 512, 1024, 2048\}$ with stride 64, capped at 20 positions per sequence
- Saves results to `data/entropy_results_{model}.csv`, checkpointing after every (domain, N) cell

The notebooks were run on the NYU HPC cluster. On a single A100 GPU, each notebook takes approximately 4 hours.

### 3. Run the analysis

Open `entropy_analysis.ipynb`. It reads from `data/entropy_results_combined.csv`, fits exponential and power-law decay models, and writes figures to `analysis/`.

---

## Key parameters

| Parameter | Value |
|---|---|
| Models | Llama-2-7B (float16), Mistral-7B-v0.1 (float16), Qwen3-8B (bfloat16) |
| Domains | Wikipedia (WikiText-103), Fiction (Gutenberg English), Code (CodeAlpaca-18k Python) |
| Sequences per domain | 200, each ≥ 5,000 tokens |
| Context lengths N | 16, 64, 256, 512, 1024, 2048 |
| Stride | 64 tokens |
| Max positions per sequence | 20 |

---

## Data

All cached token files and raw results are included in `data/`. If you want to re-tokenize from scratch, delete the `.json` files and re-run the evaluation notebooks from the data-loading cells.
