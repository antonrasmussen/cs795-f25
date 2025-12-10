# TinyLLM Phase 2 (Phi-3-mini) — GPTQ vs AWQ → GGUF

This repo gives you **starter scripts** and a **Colab notebook** to:
- Quantize **microsoft/Phi-3-mini-4k-instruct** with **GPTQ (4-bit)** and **AWQ (4-bit)**
- Evaluate **PPL (WikiText-103/2)** and **mini MMLU/ARC/GSM8K**
- Measure **CPU perf**: peak RAM, latency (ms/token), throughput (tok/s)
- Export **GGUF** and run locally with **llama.cpp** on CPU
- Produce a **comparison table + plots** and a **≤2-page report**
  
> If Phi-3 conversion to GGUF is fussy, swap to **Mistral-7B-Instruct** for the GGUF demo. The methodology is identical.

## Quick Start (Colab)
1. Open `colab/VertexAI_Colab_TinyLLM.ipynb` in Google Colab.
2. Run the **Environment** cell to install deps.
3. Run `scripts/quantize_gptq.py` and `scripts/quantize_awq.py`.
4. Run `scripts/evaluate.py` three times (fp16|gptq|awq) to populate `results/metrics.csv`.
5. Build `llama.cpp`, run `scripts/export_to_gguf.py`, then test `./main` on CPU.
6. Open `results/metrics.csv` to see the table, plus generated plots in `results/charts/`.

## CLI
```bash
# GPTQ 4-bit
python scripts/quantize_gptq.py --model microsoft/Phi-3-mini-4k-instruct --out models/phi3mini-gptq-4bit

# AWQ 4-bit
python scripts/quantize_awq.py  --model microsoft/Phi-3-mini-4k-instruct --out models/phi3mini-awq-4bit

# Baseline FP16 cache (for eval speed)
python -m transformers.models.auto.convert_slow_tokenizer --model microsoft/Phi-3-mini-4k-instruct  # optional

# Evaluate (produces metrics.csv and charts)
python scripts/evaluate.py --which fp16 --model microsoft/Phi-3-mini-4k-instruct
python scripts/evaluate.py --which gptq --model_dir models/phi3mini-gptq-4bit
python scripts/evaluate.py --which awq  --model_dir models/phi3mini-awq-4bit

# Export FP16 → GGUF, then quantize (inside llama.cpp) and run on CPU
python scripts/export_to_gguf.py --hf_model microsoft/Phi-3-mini-4k-instruct --out_gguf models/phi3mini-f16.gguf
# in llama.cpp:
#   ./quantize models/phi3mini-f16.gguf models/phi3mini-q4_0.gguf q4_0
#   ./main -m models/phi3mini-q4_0.gguf -p "Explain quantization in 2 sentences." -n 128 --threads $(nproc)
```

## Folder layout
```
tinyllm-phase2/
  README.md
  requirements.txt
  env.yml
  colab/
    VertexAI_Colab_TinyLLM.ipynb
  data/
    calib_wikitext.txt                # auto-created on first run
    eval_mmlu_50.jsonl                # auto-created on first run
    eval_arc_easy_50.jsonl            # auto-created on first run
    eval_gsm8k_25.jsonl               # auto-created on first run
  models/                             
    phi3mini-fp16/                    # optional cache
    phi3mini-gptq-4bit/               # saved by quantize_gptq.py
    phi3mini-awq-4bit/                # saved by quantize_awq.py
    phi3mini-f16.gguf                 # created by export_to_gguf.py
    phi3mini-q4_0.gguf                # created by llama.cpp's quantize tool
  scripts/
    quantize_gptq.py
    quantize_awq.py
    evaluate.py
    export_to_gguf.py
    make_plots.py
  results/
    metrics.csv
    charts/
      accuracy_vs_size.png
      speed_vs_ram.png
```

## Notes
- Pin versions in `requirements.txt` for reproducibility.
- CPU runs will be slower; keep eval subsets small (50/50/25).
- If `awq` wheel is unavailable on your OS, install from source per the GitHub README.
- Vertex AI is **optional** here; you can still use Colab CPU. The notebook includes an optional section to store artifacts in GCS and to use a Vertex AI CPU VM if desired.
