#!/usr/bin/env python
import argparse, json, os, sys, time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

def get_calib_texts(n=1024):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:2048]")
    texts = [x["text"] for x in ds if x["text"] and x["text"].strip()]
    return texts[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--out", default="models/phi3mini-gptq-4bit")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--desc", default="gptq-4bit")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("[GPTQ] Loading FP16 model & tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")

    calib_texts = get_calib_texts(1024)
    examples = [{"input_ids": tok(t, return_tensors="pt")["input_ids"]} for t in calib_texts]

    print(f"[GPTQ] Quantizing to {args.bits}-bit, group_size={args.group_size} ...", flush=True)
    qcfg = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        damp_percent=0.01,
        desc_act=True,   # act-order
    )
    qmodel = AutoGPTQForCausalLM.from_pretrained(base, quantize_config=qcfg)
    qmodel.quantize(examples)

    print(f"[GPTQ] Saving quantized model to {args.out} ...", flush=True)
    qmodel.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    with open(os.path.join(args.out, "quant_info.json"), "w") as f:
        json.dump({"method": "gptq", "bits": args.bits, "group_size": args.group_size}, f, indent=2)
    print("[GPTQ] Done.")

if __name__ == "__main__":
    main()
