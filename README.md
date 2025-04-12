# Improving LLM Safety and Helpfulness using SFT and DPO 🤖🛡️

### Description 🖋️

This repository contains the implementation for the paper titled  
**"Improving Language Model's Safety and Helpfulness using SFT and DPO: A Study on OPT-350M"**.

The project explores two core alignment techniques—**Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)**—to enhance the safety and helpfulness of an open-source language model (OPT-350M).  
It systematically compares these techniques using the **Anthropic HH-RLHF dataset**, and evaluates them via a reward model on **harmlessness** and **helpfulness** criteria.

### Pipeline Overview 🔁

This repo implements the full experimental pipeline from the paper:

1. **Supervised Fine-Tuning (SFT)** on chosen responses.
2. **Direct Preference Optimization (DPO)** using preference pairs.
3. **Evaluation** of all models (Base, SFT, DPO, SFT+DPO) using:
   - Reward scoring from `OpenAssistant/reward-model-deberta-v3-large-v2`
   - Custom metrics: Harmlessness Rate (HmR), Helpfulness Rate (HpR), and Combined Alignment Score (CAS)

### Features 🌟

- Clean SFT and DPO training using **TRL (Transformer Reinforcement Learning)** library.
- Full data preprocessing for `Anthropic/hh-rlhf` dataset.
- PEFT (LoRA) support for DPO to enable low-resource fine-tuning.
- Reward model-based evaluation and metric computation.
- Pre-generated JSON responses and classification outputs.
- Matplotlib visualizations and detailed result analysis included.

### Prerequisites 📦

- Python 3.9+
- GPU with ≥12GB VRAM (A100 preferred for full training), Google Collab would work as well.
- Libraries: `transformers`, `trl`, `peft`, `datasets`, `torch`, `matplotlib`, `tqdm`
- OpenAssistant reward model: `OpenAssistant/reward-model-deberta-v3-large-v2`

### How to Use 🛠️

This project is implemented in two main Jupyter notebooks:

1. **`main.ipynb`**

   - Handles data loading, formatting, and training for both SFT and DPO.
   - Uses TRL and PEFT (LoRA) for efficient fine-tuning.

2. **`eval.ipynb`**
   - Generates responses from all 4 models (Base, SFT, DPO, SFT+DPO) on a set of 100 prompts.
   - Uses the reward model `OpenAssistant/reward-model-deberta-v3-large-v2` to evaluate each response.
   - Computes three metrics:
     - **HmR (Harmlessness Rate)** – % of responses considered safe.
     - **HpR (Helpfulness Rate)** – % of responses considered helpful.
     - **CAS (Combined Alignment Score)** – average of HmR and HpR.
   - Outputs all results as JSON and plots evaluation graphs.

Each notebook is designed to be run sequentially with clear cell annotations and outputs.

### Outputs 📊

- Trained SFT and DPO models (via PEFT - LoRA) on OPT-350M.
- Generated responses for each model on 50 harmful and 50 helpful prompts.
- JSON files containing prompt-response pairs and their reward model scores.
- Classification of responses into harmful/harmless and helpful/unhelpful.
- Computed metrics:
  - **HmR** – Harmlessness Rate
  - **HpR** – Helpfulness Rate
  - **CAS** – Combined Alignment Score
- Matplotlib-based visualizations for metric comparison and average reward scores.
- Final summary of results in tabular and graphical format, as used in the research paper.

### Acknowledgments 🙌

This research work focused on aligning small-scale language models using SFT and DPO.  
Special thanks to **Hugging Face**, **TRL**, and **OpenAssistant** communities for open tools and datasets that made this work possible.

> By Piyush Pant ( पियूष पंत )
