---
layout: cover
background: ./images/transformer_cover_bg.jpg
class: text-center
---


# Attention Is All You Need
### *CS795 Presentation 1*


Anton Rasmussen
<br>
Old Dominion University
<br>
Fall 2025


<div class="mt-8 text-lg opacity-80">
<i>Exploring the architecture that transformed Natural Language Processing and paved the way for LLMs.</i>
</div>

<!-- Good evening, everyone. My name is Anton Rasmussen, and tonight we're diving into one of the most influential papers in modern AI: 'Attention Is All You Need.' -->

---

## 📖 The Paper
**Vaswani, A., Shazeer, N., Parmar, N., et al.** 2017. *Attention is All You Need.* NeurIPS 2017.  
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

<br>

➡️ Moved away from RNNs and LSTMs

➡️ Laid foundation for LLMs like GPT

<!-- Published in 2017 by researchers at Google, this paper introduced the Transformer, an architecture that completely changed the field of Natural Language Processing. 

It moved us away from the sequential, pinhole-like processing of RNNs and LSTMs and laid the foundation for the large language models, like GPT, that we see everywhere today. 

Recurrent Neural Networks (RNNs) are a type of neural network designed to process sequential data by incorporating feedback loops, allowing them to maintain "memory" of past information to inform current predictions. 

Long Short-Term Memory (LSTM) networks are a specialized, advanced type of RNN that use internal "gates" and a "memory cell" to overcome the vanishing gradient problem, which hinders standard RNNs from learning long-term dependencies in data.  

GPT, or Generative Pre-trained Transformer, is a type of artificial intelligence neural network that analyzes prompts (text, images, or sound) to predict and generate the most appropriate response

-->

---

## 🎯 Objectives

<br>

- Introduce the **Transformer** architecture
- Explain **self-attention** and **multi-head attention**
- Highlight **key results** and why this paper changed NLP
- Discuss **strengths, limitations, and future directions**

<!-- Tonight's goal is to understand why this paper was revolutionary. We'll start by introducing the Transformer architecture, zoom in on its core mechanism—self-attention and multi-head attention—then explore key results, strengths, and limitations, finishing with its lasting impact and future directions. By the end, you should clearly see why the authors confidently claimed that 'Attention is all you need.' -->

---

## 🏗️ Motivation

<br>

- RNNs and LSTMs process input **sequentially** → slow and hard to parallelize
- Struggle with **long-range dependencies**
- CNNs help but require very deep stacks for global context
- **Goal:** Design an architecture that is faster, scalable, and captures dependencies across entire sequence efficiently

<img src="./images/rnn_vs_transformer_parallelization.png" class="w-1/2 mx-auto rounded-lg shadow-lg" />


<!-- Imagine trying to understand a paragraph through a pinhole, one word at a time. By the time you reach the end, your memory of the first words is fuzzy. This is how RNNs work: slow, sequential, and prone to forgetting long-range context.

The Transformer solves this by opening the whole page at once. Instead of reading word-by-word, it sees the entire sequence simultaneously, processing everything in parallel. This makes training faster, scalable, and better at remembering early words even when the sequence is long. 

### Image: 

This slide compares RNN encoders and Transformer encoders, focusing on how they process sequences. 

**Left side – RNN Encoder:** 

- RNNs process tokens **sequentially**. 
- Each hidden state depends on the previous one, so we have to wait until all time steps are processed before obtaining the final representation of a sentence. 
- This sequential nature creates **latency bottlenecks**—parallelization is difficult because each step depends on the previous step’s output. 
- Gated variants (LSTMs, GRUs) reduce issues like vanishing gradients, but they still require many nonlinear transformations per time step, making them slower for long sequences. 
- Vanishing/exploding gradients can still be problematic, especially for very long dependencies. 

**Right side – Transformer Encoder:** 

- The Transformer encoder ingests the **entire sentence at once**, representing all tokens in parallel. 
- Using **self-attention**, each token computes its relationship with every other token in the sequence simultaneously (through scaled dot-product attention). 
- The linear projections (query, key, value) are applied in parallel across all tokens, which makes training **highly parallelizable** on GPUs/TPUs. 
- This parallelism enables faster training and inference and makes it easier to capture long-range dependencies because every token has a direct path to every other token in a single layer. 

**Takeaway:** 

- The core difference is **sequential vs. parallel processing**. 
- Transformers overcome the fundamental RNN bottleneck by enabling global context computation in O(1) sequential steps (per layer), which scales better with long sequences and large datasets.


Image source: https://data-science-blog.com/blog/2021/04/22/positional-encoding-residual-connections-padding-masks-all-the-details-of-transformer-model/

-->

---

## 💡 Key Idea

<br>

> *"Attention is all you need."*

<br>


- Use **self-attention** to relate every token to every other token in parallel
- Add **positional encodings** to preserve word order
- Leverage **multi-head attention** to capture different types of relationships simultaneously

<!-- Instead of the pinhole view, imagine writing the entire sentence on a whiteboard in a conference room. Each analyst (attention head) can look at all the words at once and decide which ones are most relevant. Positional encodings act like seat numbers to keep word order intact, and multiple heads act like different analyst teams focusing on grammar, meaning, or other relationships. -->

---

## 🏛️ Transformer Architecture

<div class="grid grid-cols-2 gap-6 items-center">

<div>

- **Encoder**: Stacks of self-attention + FFN  
- **Decoder**: Masked attention + cross-attention to encoder outputs  
- **Residual + LayerNorm** stabilize training  
- **Position-wise FFN** applied identically to each token  

</div>

<div class="flex justify-center">
  <img src="./images/transformer_architecture.png" class="w-3/4 rounded-lg shadow-lg" />
</div>

</div>

<!-- Using the same whiteboard analogy: The encoder is the team collaboratively analyzing the input sentence, word by word, in parallel. The decoder is another team that generates output, occasionally glancing back at the encoder’s whiteboard (cross-attention) to ensure its output stays grounded in the input. 


### Image:

This slide shows the **full Transformer architecture** introduced in *Attention is All You Need*, including the encoder stack (left) and decoder stack (right).

**Left side – Encoder Stack:**

* The encoder consists of **N identical layers** (typically 6).
* Each layer has two main sub-layers:

  * **Multi-Head Self-Attention:** Allows every token to attend to all other tokens in the input sequence simultaneously.
  * **Feed-Forward Network:** A position-wise fully connected network applied to each token independently.
* **Residual connections** and **Layer Normalization** (Add & Norm) wrap each sub-layer, which stabilizes training and improves gradient flow.
* Input tokens are first converted into **embeddings** and combined with **positional encodings** to retain word order before entering the first encoder layer.

**Right side – Decoder Stack:**

* The decoder also has **N identical layers**, each with three sub-layers:

  * **Masked Multi-Head Self-Attention:** Similar to the encoder’s attention but prevents attending to future tokens (causal mask), ensuring autoregressive generation.
  * **Cross-Attention:** Multi-head attention over encoder outputs, enabling the decoder to "look back" at the encoded input sequence.
  * **Feed-Forward Network:** Same as in the encoder, applied position-wise.
* Each sub-layer uses residual connections and layer normalization.
* Decoder inputs are **shifted right** so the model only has access to previously generated tokens during training--this is *only* during training (teacher forcing).

**Output Layer:**

* The final decoder representation is passed through a **linear projection** and **softmax** to produce a probability distribution over the vocabulary for the next token prediction.

**Takeaway:**

* The architecture is **fully parallelizable** on the encoder side and uses **autoregressive decoding** on the decoder side.
* The design cleanly separates **context-building (encoder)** from **sequence generation (decoder)** while leveraging attention as the primary mechanism for both intra-sequence and cross-sequence information flow.
* This modular design scales efficiently to deep stacks, enabling the Transformer to achieve state-of-the-art results across NLP tasks.


Image source: from the paper

-->


---

## 🔍 Self-Attention

<br>

1. Compute similarity scores between tokens
2. Apply softmax to get attention weights
3. Combine values into context-aware representation

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

<div class="flex justify-center">
    <img src="./images/scaled_dot_product.png" class="w-1/4 rounded-lg shadow-lg" />
</div>

<!-- Queries, Keys, and Values make self-attention work. Continuing our analogy: a Query is the question each word is asking (e.g., 'Who is it referring to?'), Keys are the labels of other words ('I am the boat'), and Values are the actual meanings. The model scores each Q-K pair, focuses on the most relevant words, and blends their Values into a context-aware representation. 

### Image:

The diagram is the **block flow** of the same computation as the scaled dot-product attention equation (with an optional mask). 

It shows the order of operations applied to $Q$, $K$, and $V$.

**Step-by-step (equation ⇄ diagram mapping):**

* **Inputs $Q, K, V$**

  * From learned linear projections of token representations (per head).
  * Typical shapes: $Q\in\mathbb{R}^{B\times L_q\times d_k}$, $K\in\mathbb{R}^{B\times L_k\times d_k}$, $V\in\mathbb{R}^{B\times L_k\times d_v}$.

* **$Q$** is a real-valued **tensor** (an array) whose **shape** is $(B, L_q, d_k)$.
* **$K$** is a real-valued **tensor** whose **shape** is $(B, L_k, d_k)$.

Think of $\mathbb{R}^{m\times n}$ (or with more factors) as “the set of all real matrices/tensors with that shape.”

### What each dimension usually means

* **$B$** — batch size (how many sequences processed together).
* **$L_q$** — length of the **query** sequence (number of query tokens/positions).
* **$L_k$** — length of the **key/value** sequence (can differ from $L_q$ in cross-attention).
* **$d_k$** — feature dimension of each **key/query** vector for a head (e.g., 64).

(You’ll often also see $V \in \mathbb{R}^{B\times L_k\times d_v}$, where $d_v$ is the value dimension.)

### How the shapes line up in attention

For a single head:

* $Q$ has shape $(B, L_q, d_k)$
* $K$ has shape $(B, L_k, d_k)$
* $V$ has shape $(B, L_k, d_v)$

Operations:

1. **Scores:** $QK^\top \rightarrow (B, L_q, L_k)$
   (dot products between each query and every key)
2. **Softmax over $L_k$:** weights $(B, L_q, L_k)$
3. **Weighted sum:** $\text{softmax}(QK^\top/\sqrt{d_k})\,V \rightarrow (B, L_q, d_v)$

### With multiple heads (what you’ll see in code)

Frameworks often keep a head axis:

* $Q \in \mathbb{R}^{B\times h\times L_q\times d_k}$,
  $K \in \mathbb{R}^{B\times h\times L_k\times d_k}$,
  $V \in \mathbb{R}^{B\times h\times L_k\times d_v}$.

Then:

* $QK^\top \rightarrow (B, h, L_q, L_k)$
* weights $\times V \rightarrow (B, h, L_q, d_v)$
* **Concat heads** on the feature axis $\rightarrow (B, L_q, h\cdot d_v)= (B, L_q, d_{\text{model}})$.

So, read “$Q \in \mathbb{R}^{B\times L_q\times d_k}$” as:
**“$Q$ is a batch of $L_q$ query vectors, each of dimension $d_k$, with real entries.”**


* **Bottom “MatMul” → $QK^\top$**

  * Computes pairwise similarity scores between each query and all keys.
  * Output $S\in\mathbb{R}^{B\times L_q\times L_k}$ (the attention score matrix).

* **Scale → $S/\sqrt{d_k}$**

  * Dividing by $\sqrt{d_k}$ keeps logits in a numerically stable range so softmax gradients don’t vanish.
  * (In code this is often fused with the matmul for efficiency.)

* **Mask (opt.) → $S' = S/\sqrt{d_k} + M$**

  * Adds a mask $M$ before softmax.
  * **Causal mask (decoder):** upper-triangular $-\infty$ to block attention to future positions.
  * **Padding mask (encoder/decoder):** $-\infty$ at padded tokens so they get zero weight.
  * Implemented as adding large negative values rather than an explicit “mask” op.

* **SoftMax → $A=\text{softmax}(S')$**

  * Row-wise over the last dimension (keys).
  * Converts scores to attention weights; each row sums to 1.
  * (Often preceded by subtracting the per-row max for numeric stability and followed by dropout.)

* **Top “MatMul” → $A V$**

  * Forms a weighted sum of value vectors to produce the attended representations.
  * Output $O\in\mathbb{R}^{B\times L_q\times d_v}$.

**Multi-head context:**

* The above pipeline runs **per head** with smaller $d_k,d_v$; outputs from all heads are concatenated and linearly projected.

**Complexity note:**

* The score matrix $S$ has size $L_q\times L_k$; computing and storing it is **quadratic in sequence length**, which is the main memory/compute bottleneck of standard attention.
* NOTE: this is with respect to sequence length $n$, not model size, and that it impacts both compute and memory.

**Takeaway:**

* The diagram is a procedural view of the same formula: **similarity (MatMul) → scaling → masking → normalization (Softmax) → value aggregation (MatMul)**, yielding context-aware token representations.


Image source: from the paper

-->

---

## 🎭 Multi-Head Attention

<div class="grid grid-cols-2 gap-6 items-center">

<div>

- **Multiple heads** learn diverse relationships
- Outputs concatenated and projected back
- Improves model expressiveness

</div>

<div>
<div class="grid grid-rows-2 gap-2 items-center">
    <img src="./images/multi_head_attention.png" class="w-1/2 mx-auto rounded-lg shadow-lg" />
    <img src="./images/transformer_attention_heads_qkv.png" class="w-4/5 mx-auto rounded-lg shadow-lg" />

</div>
</div>

</div>




<!-- Multi-head attention is like having several groups of analysts in the room, each with a different specialty. 
One group may focus on grammar, another on spatial relations, another on semantics. 
The final result combines all their insights into one richer representation. 

### Image:

These two figures depict the **same component at two zoom levels**: multi-head attention.

**Top image – Multi-Head Attention block (zoomed out):**

* Each branch runs **Scaled Dot-Product Attention** independently:

  * $\text{head}_i=\text{Attention}(Q_i,K_i,V_i)=\text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right)V_i$.
* The $h$ head outputs are **concatenated** along the feature dimension:

  * $\text{Concat}(\text{head}_1,\ldots,\text{head}_h)\in\mathbb{R}^{B\times L\times (h\cdot d_v)}$.
* A final **Linear** layer (often written $W^{O}\in\mathbb{R}^{h d_v\times d_{\text{model}}}$) maps back to the model dimension:

  * $\text{MHA}(X)=\text{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W^{O}$.
* (Not pictured: dropout, residual add, and layer norm typically wrap this block.)

**Bottom image – Per-head Q/K/V projections (zoomed in):**

* Input sequence $X \in \mathbb{R}^{B\times L\times d_{\text{model}}}$ is projected into **head-specific subspaces**.
* For head $i$:

  * $Q_i = X W_i^{Q}$, $K_i = X W_i^{K}$, $V_i = X W_i^{V}$ with $W_i^{Q}, W_i^{K} \in \mathbb{R}^{d_{\text{model}}\times d_k}$ and $W_i^{V} \in \mathbb{R}^{d_{\text{model}}\times d_v}$.
* Each head gets its **own learned matrices** $W_i^{Q}, W_i^{K}, W_i^{V}$, so different heads view the tokens through different linear lenses (different subspaces/features).
* Typical choice: $d_k=d_v=d_{\text{model}}/h$ so total compute stays comparable to a single large head.

**How they relate:**

* The **top image** shows the **entire multi-head module**—all heads run in parallel, each computes attention with its own $Q_i,K_i,V_i$; their results are concatenated and linearly mixed into the output.
* The **bottom image** shows what happens **inside one or two individual heads**—how $X$ becomes $Q_i,K_i,V_i$ using head-specific projection matrices.

**Takeaway:**

* Multi-head attention = **many smaller attentions in parallel**, each attending in a different representation subspace. This boosts expressivity and lets the model capture diverse relations (syntax, coreference, positional cues) while keeping compute balanced by splitting $d_{\text{model}}$ across heads.

Image 1 source: from the paper

Image 2 source: https://github.com/jalammar/jalammar.github.io/blob/master/images/t/transformer_attention_heads_qkv.png
-->

---

## 🧮 Positional Encoding

<br>

<img src="./images/example_of_positional_encoding_in_transformers.webp" class="w-2/3 mx-auto rounded-lg shadow-lg" />

<br>

- Encodes order with sin/cos patterns
- Allows extrapolation to longer sequences

<!-- To avoid turning a sentence into a bag of words, positional encoding adds a unique numerical 'seat number' to each token. Using sine and cosine functions, these encodings let the model infer order and even generalize to longer sequences it hasn't seen before—like giving the model a map of where each word sits in the sentence. 

### Image:

This slide explains **sinusoidal positional encoding** in the original Transformer and where it’s injected into the model.

**Left side – Where it plugs in:**

* Positional encodings are **added element-wise** to token embeddings **before** the first encoder/decoder layer.
* Shapes: embeddings $E\in\mathbb{R}^{L\times d_{\text{model}}}$, positional encodings $PE\in\mathbb{R}^{L\times d_{\text{model}}}$; input to the stack is $E+PE$.
* (Embeddings are typically scaled by $\sqrt{d_{\text{model}}}$ prior to the addition to keep magnitudes comparable.)

**Top-right – Formula (absolute, sinusoidal):**

$$
PE_{(pos,\,2i)}   = \sin\!\Big(pos/10000^{\,2i/d_{\text{model}}}\Big),\qquad
PE_{(pos,\,2i+1)} = \cos\!\Big(pos/10000^{\,2i/d_{\text{model}}}\Big)
$$

* $pos$: token position (0…$L-1$).
* $i$: dimension pair index (even/odd dims form $\sin/\cos$ pairs).
* Each pair uses a **different wavelength**; low $i$ → long wavelengths (global order), high $i$ → short wavelengths (local order).

**Middle-right – Example vector:**

* For $d_{\text{model}}=5$, the PE for position $n$ is the row $[\,\cos(n/10000^{2/5}),\; \sin(n/10000^{4/5}),\; \cos(n/10000^{6/5}),\; \sin(n/10000^{8/5}),\; \cos(n/10000^{10/5})\,]$ (ordering matches the diagram’s demo).
* Each token gets a **unique $d_{\text{model}}$-dim vector** determined solely by its position.

**Bottom-right – Addition to embeddings:**

* The model input at position $pos$ becomes $x_{pos}=e_{pos}+PE_{pos}$.
* Adding (instead of concatenating) keeps the dimension at $d_{\text{model}}$ and lets attention **see position via phases** in every feature channel.

**Why this works / properties:**

* No learned parameters; generalizes to **longer sequences** not seen in training.
* Inner products of PEs for positions $p$ and $p+k$ are functions of **offset $k$**, so attention can infer **relative distances** from absolute codes.
* Multi-frequency design gives both **global** and **local** positional sensitivity.

**Takeaway:**

* Positional encoding supplies order information that embeddings lack, by injecting a **multi-scale sinusoidal code** at each position.
* The encoder/decoder then operate on $E+PE$, enabling attention to reason about **where** tokens occur as well as **what** they are.
* Variants you may encounter: **learned** position embeddings, **RoPE** (rotary), **ALiBi**—all different ways to encode position, but the diagram depicts the original sinusoidal scheme.

Image source: https://aiml.com/explain-the-need-for-positional-encoding-in-transformer-models/

-->

---

## 📊 Results (WMT 2014)

<br>

- **English→German:** 28.4 BLEU (SOTA)
- **English→French:** 41.8 BLEU (SOTA)
- **2× faster training** vs. best RNN models
- **First SOTA model** without recurrence or convolution

<!-- The Transformer achieved state-of-the-art BLEU scores on WMT 2014, while training twice as fast as the best RNN models. This was the first top-performing model to completely eliminate recurrence and convolution. 


BLEU (BiLingual Evaluation Understudy) is an automated metric used to evaluate the quality of machine-translated text by comparing the machine-generated translation to one or more human-created reference translations

WMT 2014 is a widely used English-German dataset and is a cornerstone resource for researchers developing and evaluating machine translation (MT) systems.



-->

---

## 🔑 Strengths

<br>

- **Parallelizable** → faster training/inference
- **Handles long-range dependencies**
- **Simplicity:** no recurrence or convolution
- **Interpretability:** visualize attention weights
- Foundation for **BERT, GPT, T5, etc.**

<!-- The combination of speed, scalability, and simplicity made this architecture the backbone for modern NLP. The ability to visualize attention weights gave researchers a rare peek into what the model was 'thinking.' -->

---

## ⚠️ Limitations

<br>

- **Quadratic complexity** O(n²) → compute + memory bottleneck
- Requires **large datasets + compute**
- Fixed **positional encoding** may not generalize to longer contexts

<!-- The main drawback is cost: doubling sequence length quadruples computation and memory needs. This makes very long sequences challenging and motivates research into efficient transformers. -->

---

## 🧠 Conceptual Impact

<br>

- Sparked the **Transformer Revolution**
- Scaled with data/compute → emergence of LLMs
- Extended beyond text → Vision, Audio, Multimodal Transformers (e.g. ViT (Vision Transformer) or Whisper)

<!-- This paper triggered a paradigm shift: from text to vision, audio, and multimodal learning, the Transformer became the architecture of choice. Its scalability unlocked today's LLMs. -->

---

## 📚 Follow-up Work

<br>

- **BERT (2018):** Bidirectional pretraining
- **GPT (2018+):** Autoregressive transformers
- **T5 (2019):** Unified text-to-text
- **Efficient Transformers:** Linformer, Performer, Longformer

<!-- Follow-up research refined and extended the Transformer in multiple directions—BERT for better context, GPT for text generation, and efficient variants for longer inputs. -->

---

## 🔮 Future Directions

<br>

- Sparse/Linear Attention → reduce compute
- Long-context models (1M+ tokens)
- Multimodal fusion (text+image+audio)
- Better interpretability + safety research
- Memory-efficient attention mechanisms like FlashAttention or xFormers

<!-- The research frontier focuses on efficiency, scaling to million-token contexts, fusing modalities, and ensuring these models remain safe and interpretable. -->

---


## 🏁 Key Takeaways

<br>

- Attention enabled **parallelization + scalability**
- Paved way for **LLMs**
- Efficiency + interpretability are active research frontiers

<!-- In short: attention unlocked parallelization and scalability, paving the way for today's LLMs. But efficiency and interpretability are still open challenges. -->

---

## 🙌 Thank You!
**Questions & Discussion**

<!-- Leave this slide up for Q&A and encourage discussion. -->
