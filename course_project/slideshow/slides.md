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
Hi everyone, my name is Anton Rasmussen. This presentation is for my CS795 / CS895
Generative AI final project with Dr. Hong Qin.

This project is intentionally exploratory. It is not a polished product, but rather
a research-plus-engineering effort focused on system design decisions around deploying
large language models in sensitive domains like healthcare.

The core theme is how we move from powerful cloud-based AI systems toward smaller,
efficient, privacy-aware models that can run closer to the data, potentially on CPUs
or edge devices.
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
Large Language Models are extremely capable, but they come with high computational
and financial costs. Most modern LLM deployments assume centralized cloud inference,
persistent connectivity, and GPU availability.

Healthcare data breaks that assumption. It is highly sensitive, regulated, and often
generated continuously by personal devices like wearables or home monitors.

If inference happens only in the cloud, we introduce privacy risk, latency, dependency
on external infrastructure, and limited personalization.

This leads to the core question driving the project: can we combine the strengths of
cloud-scale training with local, efficient inference?
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
The vision is a hybrid cloud–edge architecture.

The cloud is used for what it does best: training, orchestration, versioning, and
evaluation. The edge is used for low-latency inference, privacy preservation, and
personalization.

In this model, patient data stays local. Only model updates or aggregated information
move upstream. This also opens the door to federated learning in future work.
-->

---

# 🧾 Abstract

<br>

This project investigates a hybrid cloud–edge framework for deploying Large Language Models in privacy-sensitive healthcare settings. Using Google Vertex AI as a cloud orchestration layer, I explore post-training model compression techniques and local inference strategies aimed at enabling efficient, CPU-based deployment of LLMs.  

Through a sequence of experiments—including Vertex AI model deployment, dynamic quantization, and a federated learning prototype—this work evaluates the feasibility, limitations, and future potential of edge-ready LLMs for personalized healthcare applications.

<!--
Read this slide slowly.

This abstract satisfies the formal course requirement. The key ideas are the use of
Vertex AI as a cloud orchestration layer, the exploration of post-training compression,
and the evaluation of CPU-based local inference.

Importantly, the work includes limitations and negative results, which is intentional
for an exploratory research project.
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
The project is intentionally structured in phases.

Phase I focuses on validating cloud infrastructure and deployment workflows.
Phase II focuses on model compression for edge inference.
Phase III is forward-looking work around federated and decentralized learning.

For this course, the deliverables are Phases I and II. Phase III is intentionally framed
as future research rather than unfinished work.
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
Before working with LLMs or edge inference, I needed to validate that I understood
real-world MLOps workflows.

Vertex AI provides managed infrastructure, model registry support, versioned
deployments, and both batch and online inference capabilities. This phase is about
correctness and reproducibility, not model complexity.
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
The model itself is intentionally simple. The goal is not predictive performance,
but verifying that the deployment pipeline works end-to-end.

This surfaced issues around serialization, environment alignment, and permissions,
which are all common real-world MLOps challenges.
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
This slide reflects real-world experience. Many issues had nothing to do with model
logic and everything to do with environment and tooling.

These lessons directly influenced how Phase II experiments were designed and executed.
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
Compression is not an optimization here, it is a requirement.

If edge inference is the goal, then model size, memory footprint, and CPU performance
become first-class constraints.
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
Post-training quantization was chosen first because it avoids retraining and allows
rapid experimentation.

A major constraint encountered here was library instability and compatibility issues,
which is itself an important practical finding.
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
The experiment was designed to be methodologically consistent: same prompt, same token
count, same environment, with only quantization changed.
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

This is a negative result. Dynamic quantization slightly increased model size and
worsened latency. This is an important finding rather than a failure.
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
Negative results are still results. This establishes a baseline and prevents false
assumptions about compression benefits.

It also validates that the benchmarking pipeline itself is correct.
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
Federated learning aligns naturally with healthcare privacy constraints by allowing
learning across devices without centralizing sensitive data.
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
This is a correctness check rather than a performance evaluation.

The goal is to verify that aggregation logic works before integrating it with real
models or training loops.
-->

---

# 🧠 Why FedAvg Matters Here

<br>

- Enables collaborative learning
- Preserves privacy
- Fits edge deployment
- Supports future personalization

<!--
FedAvg serves as a conceptual bridge between compressed models and decentralized learning.

It supports personalization while preserving privacy.
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
For a recorded presentation, narrate what would be shown live.

This slide exists to demonstrate practical grounding.
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
This slide emphasizes honest reflection.

Limitations are framed as informative constraints rather than failures.
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
Technical and research extensions are clearly separated.

This naturally extends toward PhD-level research.
-->

---

# 🩺 Healthcare Impact

<br>

- Keeps patient data local
- Reduces cloud dependence
- Enables personalization
- Improves trust & transparency

<!--
This slide reconnects the project back to its original healthcare motivation.

The value is not just technical performance, but trust, privacy, and patient empowerment.
-->

---

# 🙏 Acknowledgments

<br>

- Dr. Hong Qin
- CS795 / CS895 classmates
- Open-source ML community
- Google Vertex AI resources

<!--
Keep acknowledgments concise and professional.
-->

---

# ❓ Q & A

<br>

Thank you — I welcome questions and feedback.

<!--
End confidently and invite discussion.
-->
