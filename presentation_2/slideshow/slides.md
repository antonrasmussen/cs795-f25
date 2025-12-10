---
theme: default
class: lead
title: "Paper Review — LLaMA: Open and Efficient Foundation Language Models"
info: "CS795 Fall 2025 Paper Presentation 2"
author: "Anton Rasmussen"
---

# 🦙 LLaMA  
## Open & Efficient Foundation Language Models
<br>

### Touvron et al., 2023  

<br>

#### CS795 Paper Presentation #2

<br>

#### Anton Rasmussen

<!-- 
LLaMA is a milestone model family released by Meta in 2023.  

My goal is to provide a rigorous paper review that explains the algorithmic foundations, the architecture, then data and training strategy, and the empirical results.
 -->
---

# 🏗️ Motivation  

<br>

### Why LLaMA Matters

<br>

- LLM research dominated by *proprietary* models  
- Lack of reproducibility and transparency  
- Extremely high compute barriers  
- Limited access for healthcare, academic, and resource-constrained researchers  
- **Goal:** Build SOTA models trained *only on public data*  
- **Impact:** Democratize LLM research


<!-- 
LLaMA is not just a model release -- it is a political and scientific stance about open science.

This is especially relevant for fields like healthcare, where access and reproducibility are major equity issues.
 -->
---

# 📚 Pre-Training Data Strategy  
### *Key innovation: high-quality, public-only corpora*  

<br>


<div class="grid grid-cols-2 gap-6 items-center">

<div>


- **1.4T tokens total**  
- **100% public datasets**
- Heavy filtering, deduplication, quality scoring
- Data mixture:
  - **67%** CCNet CommonCrawl  
  - **15%** C4  
  - **4.5%** GitHub  
  - **4.5%** Wikipedia  
  - **4.5%** Books (Gutenberg + Books3)  
  - **2.5%** arXiv LaTeX  
  - **2%** StackExchange  


</div>

<div class="flex justify-center">
  <img src="./images/table_1.png" class="w-90 rounded shadow" />
</div>

</div>

<!-- 
LLaMA disproves the belief that SOTA models require proprietary datasets.

This stance is one of the most important contributions of the paper.
 -->
---

# ⚙️ Architecture Overview  
### Transformer with targeted modifications  

<div class="grid grid-cols-2 gap-8 items-start">

<div>

- **Pre-normalization** (RMSNorm)  
- **SwiGLU activation** (↑ training stability, ↑ efficiency)  
- **Rotary positional embeddings (RoPE)**  
- **No absolute positional embeddings**  
- **Parameter scale:** 7B → 65B models  
- **Longer training than scaling-laws recommended**

<br>

### Why this matters  

- Reduces inference cost  
- Enables small models to outperform large ones  
- Supports running 13B on a single GPU

</div>

<div class="flex justify-center">
  <img src="./images/llama_model_card.png" class="w-80 rounded shadow" />
</div>

</div>


<!-- 
Pre-normalization (RMSNorm)
Instead of normalizing activations after each transformer block, LLaMA normalizes inputs before they enter the block.
This stabilizes very deep networks and avoids gradient explosion, which is especially important when training long sequences across trillions of tokens.

The benefit is more reliable training without extra compute, which contributes to LLaMA’s efficiency.

SwiGLU activation
SwiGLU is a gated activation function that’s more expressive than ReLU but uses parameters efficiently.

The key advantage is that it improves the model’s ability to learn complex relationships without significantly increasing model size.
This contributes directly to LLaMA’s high performance-per-parameter.

Rotary positional embeddings (RoPE)
RoPE encodes positional information by rotating embeddings in a way that preserves relative distance between tokens.
Unlike fixed positional embeddings, RoPE generalizes better to longer contexts and improves attention patterns across long sequences.
This allows LLaMA to maintain strong performance even as context length grows.

No absolute positional embeddings
LLaMA removes learned position vectors entirely.
This reduces parameter count and avoids issues where absolute positions limit generalization.
Combined with RoPE, this choice results in lighter models that still handle sequence structure very well.

Parameter scale: 7B → 65B
The authors intentionally cover a wide range of sizes so the models fit different inference budgets.
The 7B and 13B models, in particular, are optimized to run on a single modern GPU while still performing competitively with much larger closed-source models.


Instead of following Chinchilla’s “optimal” token-to-parameter ratio, LLaMA trains its smaller models on more tokens than expected.
This is an important departure from the literature: it shows that smaller models continue improving well beyond what scaling laws predicted.
This is a major reason LLaMA-13B can outperform much larger models like GPT-3.
 -->
---

# ⚡ Training Optimization  
### Efficient Implementation  


- FlashAttention-style memory optimizations  
- Activation checkpointing  
- Model + sequence parallelism  
- Overlapping compute/communication  
- Training speed: **380 tokens/sec/GPU**  
- 65B model trained on **2048 A100 GPUs for 21 days**

<!-- 
So, why do these engineering optimizations matter?

They reduce hardware requirements for future researchers who want to replicate or extend the work. -->

---

# 📈 Training Dynamics  
### Loss Curves Across Model Sizes  

<div class="grid grid-cols-2 gap-6 items-center">

<div>

**Observation:**  
- All models continue improving past the 300B–500B token range  
- Contradicts Chinchilla-scaling expectations  
- Supports the authors’ claim that *small models benefit from long training*

</div>

<div class="flex justify-center">
  <img src="./images/training_loss_per_token.png" class="w-90 rounded shadow" />
</div>

</div>


<!-- 
This figure plots the training loss against the number of tokens seen for all four LLaMA model sizes. One of the most important observations is that all models continue improving well beyond the 300–500 billion token range where previous scaling laws, like Chinchilla’s, predicted diminishing returns.

What’s particularly interesting is that the 7B and 13B models keep showing steady loss reductions, even close to a trillion tokens. That’s surprising because traditional scaling laws suggested that smaller models should saturate earlier. Instead, these curves show that smaller models benefit significantly from being trained longer, which helps explain how LLaMA-13B manages to rival much larger models like GPT-3.

The 33B and 65B models follow the expected trend--lower loss overall due to more capacity; but, what matters here is the shape of the curves: none of them flatten prematurely. This supports the authors’ claim that the field may have underestimated the value of long training runs, especially for resource-efficient model sizes.
 -->
---

# 🧠 Benchmark Results

<div class="grid grid-cols-2 gap-6 items-center">

<div>



LLaMA excels across:

- ✔️ **Common Sense Reasoning**  
- ✔️ **Closed-book QA**  
- ✔️ **Reading comprehension**  
- ✔️ **Code generation**  
- ✔️ **Math reasoning (with sampling)**  

**Key headline result:** 

<br>

👉 *LLaMA-13B outperforms GPT-3 (175B) on most benchmarks while being over 10× smaller.*  



</div>

<div class="flex justify-center">
  <img src="./images/figure_2.png" class="w-90 rounded shadow" />
</div>

</div>



<!-- 
The main contribution here is **efficiency** -- matching or beating much larger proprietary models.
 -->
---

# 🧩 Common Sense Reasoning  


- LLaMA-65B outperforms Chinchilla-70B on nearly all tasks  
- LLaMA-13B ≈ GPT-3 175B, occasionally beating it  
- Strongest performance seen on HellaSwag & PIQA  

**Interpretation:**

<br>

Efficient training + public data still yields competitive world knowledge.

<!-- 
These tasks—BoolQ, PIQA, SIQA, HellaSwag--are widely used to evaluate grounded reasoning and consistency.
 -->
---

# ❓ Closed-Book QA  


- LLaMA-65B achieves **state-of-the-art EM accuracy** on TriviaQA and NaturalQuestions  
- Small models (7B, 13B) remain competitive  
- Demonstrates strong internal knowledge representation

<br>

<div class="grid grid-cols-2 gap-6">
<div>

### NaturalQuestions  
- 65B: **31.0% (1-shot)**  
- PaLM-540B: **29.3%**


  <img src="./images/table_4.png" class="w-50 rounded shadow" />

</div>
<div>

### TriviaQA  
- 65B: **68.2% (0-shot)**  
- Chinchilla-70B: **55.4%**

  <img src="./images/table_5.png" class="w-50 rounded shadow" />

</div>
</div>

<!-- 
These results validate the the authors’ claim:  
*High-quality public data can rival proprietary corpora.*
 -->
---

# 📘 Reading Comprehension  

<div class="grid grid-cols-2 gap-6 items-center">

<div>

- LLaMA-65B nearly matches PaLM-540B  
- Outperforms GPT-3 on RACE-middle and RACE-high  
- Shows strong contextual reasoning


</div>

<div class="flex justify-center">
  <img src="./images/table_6.png" class="w-90 rounded shadow" />
</div>

</div>


<!-- 
Note that RACE requires deeper reading and inference compared to simple QA.
 -->
---

# 🧮 Mathematical Reasoning  


<div class="grid grid-cols-2 gap-6 items-center">

<div>



- Without fine-tuning on math data, LLaMA-65B:
  - Outperforms Minerva-62B on GSM8K (majority voting)
  - Shows competitive MATH scores  
- Demonstrates emergent chain-of-thought capabilities

</div>

<div class="flex justify-center">
  <img src="./images/table_7.png" class="w-70 rounded shadow" />
</div>

</div>


<!-- 
It should be noted that this result is surprising because Google’s Minerva *was* explicitly fine-tuned on math corpora.
 -->
---

# 💻 Code Generation  


<div class="grid grid-cols-2 gap-6 items-center">

<div>

- LLaMA-33B and 65B outperform LaMDA-137B  
- LLaMA-65B ≈ PaLM-62B on HumanEval/MBPP  
- Strong performance despite *no code-specific fine-tuning*


</div>

<div class="flex justify-center">
  <img src="./images/table_8.png" class="w-70 rounded shadow" />
</div>

</div>


<!-- 
This is one of the most practically impactful results.  
Point out that code tasks require structured reasoning, syntax control, and long-range memory.
 -->
---


# 🌐 MMLU: Mixed-domain Knowledge  

<div class="grid grid-cols-2 gap-6 items-center">

<div>


- LLaMA-65B: **63.4%**
- Behind Chinchilla-70B (67.5%) and PaLM-540B (69.3%)
- Authors explain the gap: **limited book/academic data**

**Critical Interpretation:**  
The model underperforms where high-quality curated text matters most.


</div>

<div class="flex justify-center">
  <img src="./images/table_9.png" class="w-90 rounded shadow" />
</div>

</div>



<!-- 
This is the single area where LLaMA is clearly weaker, and it offers a natural entry point into critical analysis.
 -->
---

# 🧪 Bias, Toxicity, & Safety  


<div grid="~ cols-2 gap-6">
<div>

### Findings
- Toxicity ↑ with model size (RealToxicityPrompts)
- Moderate to high bias in religion, gender, SES categories
- LLaMA-65B performs better than GPT-3 in TruthfulQA, but still hallucinates
- Bias patterns reflect heavy reliance on CommonCrawl

<br>

### Interpretation
Open models enable auditability — but also distribute risks more broadly.

</div>

<div class="flex flex-col items-center space-y-4">
  <img src="./images/table_11.png" class="w-50 rounded shadow" />
  <img src="./images/table_12.png" class="w-50 rounded shadow" />
</div>
</div>

<!--
Here we see the paper’s analysis of bias and toxicity across LLaMA model sizes. The first key observation is that toxicity increases as models get larger. The top figure shows higher RealToxicityPrompts scores for the 65B model compared to smaller versions, meaning the model is more likely to produce harmful or offensive content.

Second, the bottom figure shows bias evaluations across religion, gender, and socioeconomic categories. LLaMA exhibits moderate to high bias in several of these dimensions, and the patterns closely mirror what we’d expect from CommonCrawl-heavy training data — which suggests that the dataset’s distribution strongly shapes the model’s social biases.

Interestingly, LLaMA-65B performs better than GPT-3 on TruthfulQA, which evaluates factual consistency and susceptibility to common misconceptions. But even with this improvement, hallucinations remain an issue.

The broader takeaway is that openness cuts both ways: public release allows independent auditing — which is a major scientific benefit — but it also increases the risk that harmful outputs can be deployed without strong guardrails. This tension between transparency and safety is a central theme in evaluating open LLMs.
-->

---

<div class="text-[0.9rem] leading-tight">

# 🔍 Critical Evaluation  

### Strengths  

- **High performance with public data — major contribution**  
- **Efficient architecture and training strategy**  
- Open-source research values  
- Enables reproducibility, auditability, downstream instruction tuning

<br>

### Limitations  

- Underperforms on knowledge-dense tasks (MMLU)  
- Some benchmarks show instability (SIQA variance)  
- Significant carbon cost of training (Table 15)  
- Bias & toxicity not fully addressed

<br>

### Methodological Notes 

- Longer training contradicts Chinchilla scaling laws → suggests the field has more to learn about scaling  

</div>

<!--
For strengths: LLaMA’s biggest contribution is proving that state-of-the-art performance is achievable using only public data. The architecture and training pipeline are carefully optimized, and because the models are open, they support real reproducibility and have fueled a huge wave of downstream research.

For limitations: The model still struggles on knowledge-dense tasks like MMLU, shows instability on some benchmarks, and has notable carbon and safety costs. Bias and toxicity remain unresolved issues.

Methodologically: The fact that smaller models keep improving beyond expected saturation points challenges existing scaling laws, suggesting our understanding of optimal training regimes is still incomplete.

-->

---

# 🩺 Why This Paper Matters for Healthcare AI  (my area of interest)

- Demonstrates that **open, transparent models** can match proprietary systems  
- Supports healthcare goals of fairness, transparency, reproducibility  
- Enables smaller institutions to run 7B/13B models locally  
- Facilitates domain-specific tuning (clinical notes, biomedical QA)

---

# 📝 Summary & Takeaways

- LLaMA proves **public data + efficient training** can achieve SOTA results  
- 13B model rivals or exceeds GPT-3 (175B)  
- 65B model competes with PaLM-540B and Chinchilla  
- Architecture is incremental but highly optimized  
- A landmark for **open LLM research** and scientific accessibility  
- Critical limitations include MMLU performance and safety concerns  

---

# ❓ Q&A  
### Anticipated Questions:

- *How does LLaMA differ from GPT-3 architecturally?*  
- *Does public data introduce fairness/safety tradeoffs?*  
- *Why do the authors challenge Chinchilla scaling laws?*  
- *How would you replicate this work on smaller hardware?*  
- *What is LLaMA’s role in the modern LLM ecosystem (e.g., LLaMA-2, LLaMA-3)?*

<!-- 
❓ How does LLaMA differ from GPT-3 architecturally?

LLaMA uses several efficiency-focused modifications such as RMSNorm pre-normalization, SwiGLU activations, and rotary positional embeddings, whereas GPT-3 uses older architectural defaults. These choices improve training stability and reduce parameter overhead while enhancing generalization to long sequences. As a result, LLaMA achieves comparable or better performance at much smaller parameter scales.

❓ Does public data introduce fairness or safety tradeoffs?

Yes—public data, especially CommonCrawl, contains societal biases and toxic language, which LLaMA inevitably inherits. However, using public data also improves transparency because researchers can directly inspect and audit the training sources. This tradeoff highlights the need for ongoing mitigation strategies in open-source models.

❓ Why do the authors challenge Chinchilla scaling laws?

The scaling laws predicted that smaller models should saturate after a certain token budget, but LLaMA’s loss curves show continued improvement far beyond those thresholds. This suggests that the "optimal" token-to-parameter ratio might not generalize universally across architectures. The authors argue that extended training can unlock more performance from smaller, efficient models.

❓ How would you replicate this work on smaller hardware?

LLaMA’s architecture is intentionally optimized for efficiency, so the 7B and 13B models can run inference on a single modern GPU. For training, one could adopt similar techniques—activation checkpointing, FlashAttention, and model parallelism—to reduce memory costs. Full replication at LLaMA scale remains expensive, but partial reproductions and fine-tuning are accessible.

❓ What is LLaMA’s role in the modern LLM ecosystem?

LLaMA became the foundation for a wave of open-source models, ultimately leading to LLaMA-2 and LLaMA-3 as well as community derivatives like Alpaca, Vicuna, and many others. It shifted the ecosystem toward transparency and democratized access to high-performance LLMs. Its release changed industry norms by proving that public data and efficient training pipelines can rival closed corporate models.

 -->
