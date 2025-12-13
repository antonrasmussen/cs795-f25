---
theme: default
class: lead
title: "From Vertex AI to Edge-Ready LLMs"
info: "CS795 / CS895 Final Project Presentation"
author: "Anton J. Rasmussen"
---

# From Vertex AI to Edge-Ready LLMs  
## A Hybrid Cloud–Edge Framework for  
## Privacy-Preserving Healthcare Inference

<br>

**Anton J. Rasmussen**  
Department of Computer Science  
Old Dominion University  

<br>

CS795 / CS895 — Generative AI  
Instructor: Dr. Hong Qin  
Fall 2025

<!--
Introduce yourself, the course, and frame this as a research + engineering project.
Emphasize that this work is exploratory and foundational, not a polished product.
-->

---

# 🎯 Project Motivation

<br>

### The Problem

- Large Language Models are **powerful but expensive**
- Most healthcare data is **sensitive and regulated**
- Cloud-only inference:
  - Introduces privacy risk
  - Adds latency
  - Limits personalization

<br>

### Core Question

> *Can we combine cloud-scale training with efficient, local inference for healthcare AI?*

<!--
Explain the tension between LLM capability and healthcare constraints.
Stress that privacy and deployment feasibility are the core drivers.
-->

---

# 🧠 Vision

<br>

### Big Picture

- **Train & orchestrate** models in the cloud (Vertex AI)
- **Compress & deploy** models locally (CPU / edge devices)
- **Keep patient data local**
- Enable future **federated or collaborative learning**

<br>

This project explores the **foundations** needed to make that possible.

<!--
Make it clear this is a systems-level vision, not just a single experiment.
Mention that later phases build on these foundations.
-->

---

# 🧾 Abstract

<br>

This project investigates a hybrid cloud–edge framework for deploying Large Language Models in privacy-sensitive healthcare settings. Using Google Vertex AI as a cloud orchestration layer, I explore post-training model compression techniques and local inference strategies aimed at enabling efficient, CPU-based deployment of LLMs.  

Through a sequence of experiments—including Vertex AI model deployment, dynamic quantization, and a federated learning prototype—this work evaluates the feasibility, limitations, and future potential of edge-ready LLMs for personalized healthcare applications.

<!--
Read this more slowly.
This slide satisfies the formal “abstract” requirement.
-->

---

# 🧱 Project Structure

<br>

### Three Phases

1. **Phase I — Foundations**
   - Vertex AI deployment & MLOps workflow

2. **Phase II — Compression**
   - Quantization experiments for edge inference

3. **Phase III — Collaboration (Future)**
   - Federated learning & decentralized intelligence

<br>

This course project focuses primarily on **Phases I and II**.

<!--
Explain that Phase III is forward-looking and intentionally incomplete.
This shows research maturity rather than scope creep.
-->

---

# ⚙️ Phase I: Vertex AI Foundations

<br>

### Goal

Establish an **end-to-end cloud workflow** for training, registering, and deploying models.

<br>

### Why Vertex AI?

- Managed MLOps
- Model registry & versioning
- Batch + online inference
- Scales to future LLM workloads

<!--
Tie this explicitly to real-world MLOps practices.
Mention that this was necessary before touching LLMs.
-->

---

# ☁️ Vertex AI Workflow

<br>

### Implementation Steps

1. Train a `RandomForestClassifier` on the Iris dataset
2. Serialize model artifacts
3. Upload to Vertex AI Model Registry
4. Deploy to a live endpoint
5. Execute online & batch predictions

<br>

### Key Takeaway

> This validated the **cloud-side foundation** needed for later LLM experimentation.

<!--
Emphasize correctness and reproducibility over model complexity.
-->

---

# ⚠️ Lessons from Phase I

<br>

### Challenges Encountered

- Serialization mismatches
- Python / scikit-learn version conflicts
- Limited access to Cloud Logging
- IAM permission constraints

<br>

### What This Taught Me

- Real-world MLOps is fragile
- Reproducibility matters
- Version alignment is critical

<!--
This is a strong reflection slide.
Explicitly say that these failures informed later design decisions.
-->

---

# 🔧 Phase II: Why Compression Matters

<br>

### Problem

LLMs are:

- Too large for CPUs
- Too slow for embedded systems
- Too expensive for continuous cloud inference

<br>

### Solution Space

- Post-training quantization
- Mixed precision
- Model pruning
- Efficient runtimes (e.g., `llama.cpp`)

<!--
Position compression as a necessity, not an optimization.
-->

---

# 📦 Compression Strategy

<br>

### Initial Focus

- **Post-training quantization**
- No retraining required
- Lower barrier to experimentation

<br>

### Techniques Explored

- PyTorch Dynamic INT8 Quantization
- GPTQ / AWQ (partially supported)
- GGUF export pipeline (planned)

<!--
Briefly mention library instability as a real-world constraint.
-->

---

# 🧪 Quantization Experiment

<br>

### Setup

- Model: `distilgpt2`
- Environment: CPU-only (Colab)
- Metric: Latency, size, compression ratio

<br>

### Method

- Baseline FP32 inference
- Apply dynamic INT8 quantization
- Compare performance

<!--
Stress methodological consistency here.
-->

---

# 📊 Quantization Results

<br>

| Model | Latency (32 tokens) | Size (MB) | Speedup | Compression |
|------|---------------------|-----------|----------|-------------|
| FP32 | 1.001 s | 312.5 | 1.00× | 1.00× |
| INT8 | 1.188 s | 349.3 | 0.84× | 0.89× |

<br>

### Key Observation

> Dynamic quantization **did not help** for small models.

<!--
Pause here.
Explicitly state that negative results are still results.
-->

---

# 📉 Interpretation

<br>

### What This Means

- Small models may not benefit from naive quantization
- Overhead can outweigh gains
- Advanced methods are required for real savings

<br>

### Why This Result Still Matters

- Establishes a **baseline**
- Validates experimental pipeline
- Guides future optimization choices

<!--
This slide demonstrates research judgment.
-->

---

# 🔁 Federated Learning Prototype

<br>

### Motivation

Healthcare data is:

- Distributed
- Sensitive
- Often cannot be centralized

<br>

### Idea

Use **federated averaging (FedAvg)** to combine model updates without sharing raw data.

<!--
Tie this back to healthcare privacy requirements.
-->

---

# 🧮 FedAvg Simulation

<br>

### Experiment

- Synthetic client updates
- Simple vector averaging
- Validates aggregation logic

<br>

### Result

```

Global update:
[1.05, 2.05, 3.05]

```

<!--
Emphasize that this is a correctness check, not performance evaluation.
-->

---

# 🧠 Why FedAvg Matters Here

<br>

- Enables collaborative learning
- Preserves privacy
- Fits edge deployment
- Supports future personalization

<!--
Position this as a conceptual bridge to future work.
-->

---

# 🧪 Demo (Optional / Recorded)

<br>

### Demonstration Options

- Vertex AI endpoint inference
- Quantization script walkthrough
- CPU-only inference benchmark
- FedAvg aggregation code

<!--
If recording, narrate what *would* be shown live.
-->

---

# 🔍 Discussion

<br>

### What Worked

- Vertex AI deployment pipeline
- Modular experimentation
- CPU-only workflows
- Reproducibility

<br>

### What Didn’t (Yet)

- Stable Phi-3 quantization
- AWQ / GPTQ compatibility
- Meaningful speedups on small models

<!--
This slide scores highly for honesty and reflection.
-->

---

# 🧭 Future Work

<br>

### Technical

- Full GPTQ / AWQ on Phi-3 or Mistral
- GGUF + `llama.cpp`
- Edge benchmarking
- Vertex AI custom jobs

<br>

### Research

- Federated fine-tuning
- Edge-RAG
- Personalized health inference

<!--
Clearly distinguish between engineering and research extensions.
-->

---

# 🩺 Healthcare Impact

<br>

- Keeps patient data local
- Reduces cloud dependence
- Enables personalization
- Improves trust & transparency

<!--
Connect back to motivation from the start.
-->

---

# 🙏 Acknowledgments

<br>

- Dr. Hong Qin
- CS795 / CS895 classmates
- Open-source ML community
- Google Vertex AI resources

<!--
Keep this short and professional.
-->

---

# ❓ Q & A

<br>

Thank you — I welcome questions and feedback.

<!--
End confidently.
-->