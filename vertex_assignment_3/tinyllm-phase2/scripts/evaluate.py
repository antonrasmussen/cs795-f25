#!/usr/bin/env python
import argparse, os, time, json, math, psutil, gc, pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

RESULTS = Path("results")
DATA = Path("data")
RESULTS.mkdir(parents=True, exist_ok=True)

def ensure_eval_slices(seed=42):
    import random, json
    random.seed(seed)
    # MMLU (hendrycks_test) small sample
    try:
        ds = load_dataset("hendrycks_test", "abstract_algebra", split="test")  # one subject as proxy
        rows = []
        for ex in random.sample(list(ds), k=min(50, len(ds))):
            rows.append({"question": ex["question"], "choices": ex["choices"], "answer": ex["answer"]})
        with open(DATA/"eval_mmlu_50.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r)+"\n")
    except Exception as e:
        pass
    # ARC-Easy
    try:
        ds = load_dataset("ai2_arc", "ARC-Easy", split="validation")
        rows = []
        for ex in random.sample(list(ds), k=min(50, len(ds))):
            rows.append({"question": ex["question"], "choices": ex["choices"]["text"], "answer": ex["answerKey"]})
        with open(DATA/"eval_arc_easy_50.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r)+"\n")
    except Exception as e:
        pass
    # GSM8K (short subset of test)
    try:
        ds = load_dataset("gsm8k", "main", split="test[:25]")
        rows = [{"question": r["question"], "answer": r["answer"]} for r in ds]
        with open(DATA/"eval_gsm8k_25.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r)+"\n")
    except Exception as e:
        pass

def ppl_on_wikitext(model, tok, which="wikitext-2-raw-v1"):
    ds = load_dataset("wikitext", which, split="validation")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt")
    input_ids = enc["input_ids"]
    stride = 2048
    lls = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, input_ids.size(1), stride)):
            begin_loc = max(i + stride - 2048, 0)
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i
            input_ids_slice = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_slice.clone()
            target_ids[:, :-trg_len] = -100
            out = model(input_ids_slice, labels=target_ids)
            lls.append(out.loss * trg_len)
    ppl = torch.exp(torch.stack(lls).sum() / end_loc).item()
    return ppl

def time_gen(pipe, prompt="Explain quantization in one paragraph.", gen_tokens=128):
    import numpy as np
    start_mem = psutil.Process().memory_info().rss
    t0 = time.time()
    out = pipe(prompt, max_new_tokens=gen_tokens, do_sample=False)
    t1 = time.time()
    end_mem = psutil.Process().memory_info().rss
    total_time = t1 - t0
    tok_s = gen_tokens / total_time if total_time > 0 else float("nan")
    ms_per_tok = (total_time / gen_tokens) * 1000.0
    peak_ram_gb = max(start_mem, end_mem) / (1024**3)
    return {"latency_ms_per_token": ms_per_tok, "throughput_tok_s": tok_s, "peak_ram_gb": peak_ram_gb}

def load_model(which, model_name=None, model_dir=None):
    if which == "fp16":
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        return model, tok
    elif which in ("gptq", "awq"):
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto")
        return model, tok
    else:
        raise ValueError("which must be fp16|gptq|awq")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["fp16","gptq","awq"])
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct", help="HF id for fp16 baseline")
    ap.add_argument("--model_dir", default=None, help="Path for quantized model")
    ap.add_argument("--wiki", default="wikitext-2-raw-v1", choices=["wikitext-2-raw-v1","wikitext-103-raw-v1"])
    args = ap.parse_args()

    ensure_eval_slices()

    model, tok = load_model(args.which, model_name=args.model, model_dir=args.model_dir)

    print("[Eval] Perplexity on", args.wiki)
    ppl = ppl_on_wikitext(model, tok, args.wiki)

    print("[Eval] Timing CPU generation (128 tokens)")
    pipe = TextGenerationPipeline(model=model, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)
    perf = time_gen(pipe)

    row = {
        "timestamp": int(time.time()),
        "method": args.which,
        "model_dir_or_name": args.model if args.which=="fp16" else args.model_dir,
        "ppl": ppl,
        "peak_ram_gb": perf["peak_ram_gb"],
        "throughput_tok_s": perf["throughput_tok_s"],
        "latency_ms_per_token": perf["latency_ms_per_token"],
    }
    csv_path = RESULTS/"metrics.csv"
    df = pd.DataFrame([row])
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print("[Eval] Wrote", csv_path)

    # simple plots
    import matplotlib.pyplot as plt
    plt.figure()
    dfp = df.copy()
    dfp.groupby("method")["ppl"].mean().plot(kind="bar", title="PPL (lower is better)")
    plt.tight_layout(); os.makedirs("results/charts", exist_ok=True)
    plt.savefig("results/charts/accuracy_vs_size.png")
    plt.close()

    plt.figure()
    dfp.groupby("method")["throughput_tok_s"].mean().plot(kind="bar", title="Throughput (tok/s)")
    plt.tight_layout()
    plt.savefig("results/charts/speed_vs_ram.png")
    plt.close()

    del model; gc.collect()
    print("[Eval] Done.")

if __name__ == "__main__":
    main()
