#!/usr/bin/env python
import argparse, json, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM

def get_calib_texts(n=1024):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:2048]")
    return [x["text"] for x in ds if x["text"] and x["text"].strip()][:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--out", default="models/phi3mini-awq-4bit")
    ap.add_argument("--w_bits", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("[AWQ] Loading model/tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoAWQForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")

    calib = get_calib_texts(1024)
    print(f"[AWQ] Calib set size: {len(calib)}")

    print(f"[AWQ] Quantizing to {args.w_bits}-bit, group_size={args.group_size} ...")
    # The exact AWQ API may vary; this uses a common call pattern.
    model.quantize(
        tokenizer=tok,
        calib_texts=calib,
        w_bits=args.w_bits,
        q_group_size=args.group_size,
        zero_point=True,
        version="GEMM",
    )

    print(f"[AWQ] Saving quantized model to {args.out} ...")
    model.save_quantized(args.out, merge_lora=False, safetensors=True)
    tok.save_pretrained(args.out)

    with open(os.path.join(args.out, "quant_info.json"), "w") as f:
        json.dump({"method": "awq", "bits": args.w_bits, "group_size": args.group_size}, f, indent=2)
    print("[AWQ] Done.")

if __name__ == "__main__":
    main()
