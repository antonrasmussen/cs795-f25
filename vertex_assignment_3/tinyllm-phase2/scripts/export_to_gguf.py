#!/usr/bin/env python
import argparse, os, subprocess, sys, pathlib, json

HELP = """
This script prepares FP16 → GGUF conversion via llama.cpp.

Steps (run inside your Colab/VM shell):
1) git clone https://github.com/ggerganov/llama.cpp
2) cd llama.cpp && make -j
3) python ./convert-hf-to-gguf.py --model <HF_DIR_OR_ID> --outfile <OUT.gguf> --outtype f16
4) ./quantize <OUT.gguf> <OUT-q4_0.gguf> q4_0
5) ./main -m <OUT-q4_0.gguf> -p "Explain quantization in 2 sentences." -n 128 --threads $(nproc)

Note: Some HF models require a local directory. If Phi-3 is problematic, try Mistral-7B-Instruct for the GGUF demo.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--llama_dir", default="llama.cpp")
    ap.add_argument("--out_gguf", default="models/phi3mini-f16.gguf")
    args = ap.parse_args()

    print(HELP)
    print("\nSuggested command:")
    print(f"python {args.llama_dir}/convert-hf-to-gguf.py --model {args.hf_model} --outfile {args.out_gguf} --outtype f16")

if __name__ == "__main__":
    main()
