## 🧭 Canon of Generative AI

### 1. **Auto-Encoding Variational Bayes (2013)**

*Kingma & Welling*

* Introduced **Variational Autoencoders (VAEs)**.
* Key innovation: the **reparameterization trick**, enabling efficient training of latent-variable generative models.
* Impact: foundational for probabilistic generative modeling, especially in scientific domains.
  📄 [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)

---

### 2. **Generative Adversarial Networks (GANs) (2014)**

*Goodfellow et al.*

* Proposed **adversarial training**: a generator vs. discriminator in a minimax game.
* GANs produced far sharper images than VAEs at the time.
* Sparked an entire subfield (DCGAN, StyleGAN, CycleGAN, etc.).
  📄 [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)

---

### 3. **Neural Autoregressive Distribution Estimation (NADE & PixelRNN/PixelCNN, 2014–2016)**

*Bengio et al., van den Oord et al.*

* Explored **autoregressive models** for density estimation and image generation.
* PixelCNN demonstrated pixel-by-pixel image generation, inspiring later diffusion approaches.
  📄 [Pixel Recurrent Neural Networks (2016)](https://arxiv.org/abs/1601.06759)

---

### 4. **Attention Is All You Need (2017)**

*Vaswani et al.*

* Introduced the **Transformer** architecture, removing recurrence entirely.
* Core mechanism: **self-attention**, scalable and parallelizable.
* Foundation for GPT, BERT, and almost every modern LLM.
  📄 [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

### 5. **Improving Language Understanding by Generative Pre-Training (GPT-1, 2018)**

*Radford et al. (OpenAI)*

* First demonstration of **pretraining a transformer on large text corpora** and fine-tuning for tasks.
* Proof that large, generative pretraining transfers across NLP tasks.
  📄 [PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

---

### 6. **StyleGAN (2019)**

*Karras et al. (NVIDIA)*

* Major leap in **image synthesis**, enabling high-quality, high-resolution, and controllable outputs.
* Introduced the “style-based” generator architecture.
* Famous for producing photorealistic but synthetic human faces.
  📄 [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)

---

### 7. **BERT (2019)** *(not purely generative but pivotal)*

*Devlin et al.*

* Bidirectional transformer, popularized **masked language modeling**.
* Though discriminative, BERT inspired hybrid approaches and clarified pretraining strategies.
  📄 [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

---

### 8. **Diffusion Models Beat GANs (2021)**

*Dhariwal & Nichol (OpenAI)*

* Revived **diffusion probabilistic models** as superior to GANs for image generation.
* Basis for **DALL·E 2, Imagen, Stable Diffusion**.
  📄 [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)

---

## 📝 Summary Timeline

* **2013** – VAEs (probabilistic latent modeling)
* **2014** – GANs (adversarial synthesis)
* **2016** – PixelCNN (autoregressive image models)
* **2017** – Transformer (attention revolution)
* **2018** – GPT-1 (generative pretraining for language)
* **2019** – StyleGAN (photorealistic image synthesis)
* **2019** – BERT (contextual pretraining, hybrid influence)
* **2021** – Diffusion (state-of-the-art image generation)

---

👉 Together, these eight papers form the **core syllabus** of generative AI. Reading them in order gives you the intellectual trajectory from *probabilistic inference → adversarial training → autoregressive modeling → attention → large-scale pretraining → controllable synthesis → diffusion dominance*.


---

## 🌱 Bonus Influential Papers in Generative AI

### 🔄 Image-to-Image & Domain Transfer

* **CycleGAN (2017)** – *Zhu et al.*

  * Enabled **unpaired image-to-image translation** (e.g., horses ↔ zebras).
  * Landmark for style transfer, domain adaptation, and creative AI.
    📄 [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)

---

### 🎨 Discrete Latents & Compression

* **VQ-VAE (2017, 2019)** – *van den Oord, Razavi et al.*

  * Introduced **vector quantized autoencoders**, enabling discrete latent representations.
  * Key stepping stone for **DALL·E** and **discrete diffusion models**.
    📄 [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)

---

### 🖼️ Text-to-Image Breakthroughs

* **DALL·E (2021)** – *Ramesh et al., OpenAI*

  * First large-scale **text-to-image transformer**, built on VQ-VAE.
  * Introduced the world to generative **multimodal art** at scale.
    📄 [arXiv:2102.12092](https://arxiv.org/abs/2102.12092)

* **CLIP (2021)** – *Radford et al., OpenAI*

  * Jointly trained on image–text pairs, aligning vision and language.
  * Became the foundation for text-guided image generation (paired with diffusion).
    📄 [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

---

### 📷 High-Fidelity Image Models

* **StyleGAN2/3 (2019–2021)** – *Karras et al.*

  * Improved realism and stability of GANs.
  * Still widely used in media and entertainment.
    📄 [StyleGAN2 arXiv:1912.04958](https://arxiv.org/abs/1912.04958)

---

### 📚 Scaling Laws & LLM Advances

* **Scaling Laws for Neural Language Models (Kaplan et al., 2020)**

  * Showed that model performance scales predictably with data, compute, and parameters.
  * Justified the move toward **GPT-3/PaLM-sized models**.
    📄 [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

* **PaLM (2022)** – *Chowdhery et al., Google*

  * 540B-parameter LLM with state-of-the-art reasoning and few-shot ability.
  * Introduced **chain-of-thought prompting**.
    📄 [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)

* **LLaMA (2023)** – *Meta AI*

  * Released efficient, open foundation LLMs.
  * Sparked the **open-source LLM ecosystem** (Alpaca, Vicuna, etc.).
    📄 [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)

---

### 🔍 Vision–Language & Multimodal Expansion

* **BLIP (2022)** – *Li et al.*

  * Advanced **image captioning** and vision–language pretraining.
  * Important for grounding multimodal models like BLIP-2 and Flamingo.
    📄 [arXiv:2201.12086](https://arxiv.org/abs/2201.12086)

* **Flamingo (2022)** – *DeepMind*

  * Large multimodal model (text + vision) with few-shot learning.
  * Precursor to models like GPT-4V.
    📄 [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)

---

## 📝 Bonus Timeline

* **2017** – CycleGAN (unpaired image translation)
* **2017–19** – VQ-VAE (discrete latents → text-to-image foundation)
* **2019–21** – StyleGAN2/3 (photorealistic synthesis)
* **2020** – Scaling Laws (theoretical justification for giant LMs)
* **2021** – DALL·E, CLIP (multimodal revolution)
* **2022** – PaLM (massive scaling + reasoning), BLIP, Flamingo
* **2023** – LLaMA (open LLMs democratized)

---

👉 Together with the **core canon list**, this bonus set shows **how the field diversified**:

* VAEs → probabilistic inference
* GANs → adversarial sharpness
* Transformers → unified architecture
* Diffusion → state-of-the-art generation
* * These bonus works → **domain transfer, multimodality, scaling, and open-source ecosystems**.

---

The following are **foundational generative AI papers** that **directly set the stage for “Agentic AI”** (systems that don’t just generate content but also *act, reason, and interact*).

Agentic AI isn’t one single invention — it builds on **generative models + reinforcement learning + tool use + multi-agent systems**. Here’s a breakdown of the most relevant *foundational papers*:

---

## 🧩 Core Generative AI Foundations Carrying into Agentic AI

### 1. **Attention Is All You Need (2017)** – Vaswani et al.

* Introduced the **Transformer**, the backbone of LLMs.
* Without it, there would be no GPT-4, Claude, LLaMA, etc.
* **Agentic AI relies on transformers for reasoning and planning**.

---

### 2. **Improving Language Understanding by Generative Pretraining (GPT-1, 2018)** – OpenAI

* Introduced **pretraining + fine-tuning paradigm**.
* Foundation for LLMs that later became *agents capable of reasoning across tasks*.

---

### 3. **Language Models are Few-Shot Learners (GPT-3, 2020)** – Brown et al.

* Showed **emergent in-context learning**.
* The “few-shot” capability is crucial for agents: they can adapt on the fly without retraining.
* Sparked the idea that **language models could serve as reasoning engines for agents**.

---

### 4. **Learning to Summarize with Human Feedback (OpenAI, 2020)**

* One of the first demonstrations of **Reinforcement Learning from Human Feedback (RLHF)**.
* Made LLMs *align with human intent*, which is essential for **agent trustworthiness and control**.
* Direct precursor to ChatGPT.
  📄 [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)

---

### 5. **Training Language Models to Follow Instructions with Human Feedback (InstructGPT, 2022)** – Ouyang et al.

* First **instruction-following LLM** trained with RLHF.
* Made models behave more like *cooperative assistants*.
* This is where the transition from *“text generator” → “agent you can direct”* begins.
  📄 [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

---

### 6. **Language Models are Few-Shot Learners at Reasoning (Chain-of-Thought, 2022)** – Wei et al.

* Introduced **chain-of-thought prompting**, letting models reason step-by-step.
* Critical for *planning and decision-making in agents*.
  📄 [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)

---

## 🌱 Early Agentic AI-Specific Papers

### 7. **Language Models are Few-Shot Learners at Tool Use (2022–2023)**

* Papers like **Toolformer (Schick et al., 2023)** and **ReAct (Yao et al., 2022)** showed how LLMs can:

  * Call external APIs, calculators, search engines.
  * Reason + Act interleaved (ReAct framework).
* These are foundational to **AI agents that browse, code, or automate workflows**.
  📄 [ReAct arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
  📄 [Toolformer arXiv:2302.04761](https://arxiv.org/abs/2302.04761)

---

### 8. **Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023, Stanford)**

* First system showing **autonomous multi-agent societies** powered by LLMs.
* Agents remembered, reflected, planned, and interacted with each other in a simulated town.
* A true **prototype of Agentic AI**.
  📄 [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)

---

## 📝 Summary — “From GenAI to AgentAI”

* **Transformers (2017)** → scalable reasoning engines.
* **GPT series (2018–2020)** → general-purpose generative pretraining.
* **RLHF (2020–2022)** → aligning LLMs with human intent.
* **Chain-of-Thought (2022)** → enabling reasoning.
* **Tool Use (2022–2023)** → giving models ways to *act*.
* **Generative Agents (2023)** → showing autonomous agent societies.

---

👉 So: **VAEs, GANs, and Diffusion** are central to *generative modeling of content*.
But **Agentic AI’s foundations come mainly from LLM + alignment papers (Transformers, GPT, RLHF, CoT, ReAct, Toolformer, Generative Agents).**

