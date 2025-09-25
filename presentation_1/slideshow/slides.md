---
# Attention Is All You Need
subtitle: CS795 Paper 1 Presentation
author: Anton Rasmussen
---

## 📖 Citation
**Vaswani, A., Shazeer, N., Parmar, N., et al.** 2017. *Attention is All You Need.* NeurIPS 2017.  
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

<!-- Good evening, everyone. My name is Anton Rasmussen, and tonight we're diving into one of the most influential papers in modern AI: 'Attention Is All You Need.'

Published in 2017 by researchers at Google, this paper introduced the Transformer, an architecture that completely changed the field of Natural Language Processing. 

It moved us away from the sequential, pinhole-like processing of RNNs and LSTMs and laid the foundation for the large language models, like GPT, that we see everywhere today. -->

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

The Transformer solves this by opening the whole page at once. Instead of reading word-by-word, it sees the entire sequence simultaneously, processing everything in parallel. This makes training faster, scalable, and better at remembering early words even when the sequence is long. -->

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

<!-- Using the same whiteboard analogy: The encoder is the team collaboratively analyzing the input sentence, word by word, in parallel. The decoder is another team that generates output, occasionally glancing back at the encoder’s whiteboard (cross-attention) to ensure its output stays grounded in the input. -->


---

## 🔍 Self-Attention

<br>

1. Compute similarity scores between tokens
2. Apply softmax to get attention weights
3. Combine values into context-aware representation

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

<!-- Queries, Keys, and Values make self-attention work. Continuing our analogy: a Query is the question each word is asking (e.g., 'Who is it referring to?'), Keys are the labels of other words ('I am the boat'), and Values are the actual meanings. The model scores each Q-K pair, focuses on the most relevant words, and blends their Values into a context-aware representation. -->

---

## 🎭 Multi-Head Attention

<br>

<img src="./images/transformer_attention_heads_qkv.png" class="w-3/5 mx-auto rounded-lg shadow-lg" />

<br>

- **Multiple heads** learn diverse relationships
- Outputs concatenated and projected back
- Improves model expressiveness

<!-- Multi-head attention is like having several groups of analysts in the room, each with a different specialty. One group may focus on grammar, another on spatial relations, another on semantics. The final result combines all their insights into one richer representation. -->

---

## 🧮 Positional Encoding

<br>

<img src="./images/example_of_positional_encoding_in_transformers.webp" class="w-2/3 mx-auto rounded-lg shadow-lg" />

<br>

- Encodes order with sin/cos patterns
- Allows extrapolation to longer sequences

<!-- To avoid turning a sentence into a bag of words, positional encoding adds a unique numerical 'seat number' to each token. Using sine and cosine functions, these encodings let the model infer order and even generalize to longer sequences it hasn't seen before—like giving the model a map of where each word sits in the sentence. -->

---

## 📊 Results (WMT 2014)

<br>

- **English→German:** 28.4 BLEU (SOTA)
- **English→French:** 41.8 BLEU (SOTA)
- **2× faster training** vs. best RNN models
- **First SOTA model** without recurrence or convolution

<!-- The Transformer achieved state-of-the-art BLEU scores on WMT 2014, while training twice as fast as the best RNN models. This was the first top-performing model to completely eliminate recurrence and convolution. -->

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
- Extended beyond text → Vision, Audio, Multimodal Transformers

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
