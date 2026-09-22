# Customer Support LLM Fine-Tuning with QLoRA

> **Parameter-efficient fine-tuning of Qwen3-0.6B for e-commerce customer support using 4-bit NF4 QLoRA, LoRA/PEFT, supervised fine-tuning, and multi-dimensional LLM evaluation.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)](https://huggingface.co/docs/peft)
[![QLoRA](https://img.shields.io/badge/QLoRA-4--bit%20NF4-green)](https://arxiv.org/abs/2305.14314)

## Overview

This project fine-tunes **Qwen3-0.6B** for realistic e-commerce customer-support scenarios using **QLoRA + LoRA/PEFT + supervised fine-tuning (SFT)**.

The goal is not only to train the model, but to demonstrate a complete LLM fine-tuning workflow:

**Data → Validation → QLoRA Training → Adapter Saving → Evaluation → Safety Testing → Base vs Fine-Tuned Comparison**

The project is designed to be **GPU-efficient, reproducible, modular, and interview-defendable**.

---

## Key Results

| Metric              |          Result |
| ------------------- | --------------: |
| Base Model          | Qwen/Qwen3-0.6B |
| Training Examples   |          **96** |
| Evaluation Examples |          **24** |
| Training Epochs     |           **2** |
| Learning Rate       |        **2e-4** |
| Quantization        |   **4-bit NF4** |
| LoRA Rank           |          **16** |
| LoRA Alpha          |          **32** |
| Evaluation Loss     |      **2.8624** |
| ROUGE-1             |      **0.1611** |
| ROUGE-2             |      **0.0306** |
| ROUGE-L             |      **0.1097** |
| Safety Pass Rate    |        **100%** |
| LLM Judge Safety    |         **5/5** |
| LLM Judge Tone      |         **5/5** |

Evaluation was performed on a held-out set of **24 examples**, while the LLM-as-a-Judge comparison covered **5 representative customer-support prompts**.

### Why these results matter

* **100% safety pass rate** on the project's sensitive-information safety checks.
* The fine-tuned model was evaluated against the base model on the **same customer-support scenarios**.
* Evaluation goes beyond training loss by combining **ROUGE, safety testing, qualitative inspection, LLM-as-a-Judge, and base-vs-fine-tuned comparison**.
* The model uses **parameter-efficient LoRA adapters** rather than full-model fine-tuning.

---

## Technical Highlights

### 1. Parameter-Efficient Fine-Tuning

Instead of updating every parameter of the base model, the project uses **LoRA adapters**.

Configuration:

```text
LoRA rank (r):       16
LoRA alpha:          32
LoRA dropout:        0.05
Target modules:      q_proj, k_proj, v_proj, o_proj
```

This significantly reduces the number of trainable parameters while adapting the model to the customer-support domain.

### 2. QLoRA + 4-bit NF4 Quantization

The base model is loaded using:

```text
4-bit quantization
NF4 quantization type
Double quantization
FP16 compute
```

This makes fine-tuning much more GPU-efficient and suitable for experimentation on limited hardware.

### 3. Supervised Fine-Tuning

The model was trained for:

```text
2 epochs
Learning rate: 2e-4
Training examples: 96
Evaluation examples: 24
```

The dataset uses structured conversational messages containing:

```text
system → user → assistant
```

---

## Evaluation Pipeline

The project evaluates the fine-tuned model from multiple perspectives rather than relying on a single metric.

### Evaluation Flow

```text
                 ┌────────────────────┐
                 │   Fine-Tuned Model │
                 └─────────┬──────────┘
                           │
             ┌─────────────┼─────────────┐
             ↓             ↓             ↓
        Eval Loss        ROUGE       Safety Tests
             │             │             │
             └─────────────┼─────────────┘
                           ↓
                  Qualitative Evaluation
                           │
                           ↓
                   LLM-as-a-Judge
                           │
                           ↓
                 Base vs Fine-Tuned
```

### Evaluation Methods

**Evaluation Loss**

Measures how well the fine-tuned model generalizes to the held-out evaluation set.

**ROUGE**

Measures lexical overlap between generated and reference responses.

Results:

```text
ROUGE-1     0.1611
ROUGE-2     0.0306
ROUGE-L     0.1097
ROUGE-Lsum  0.1214
```

**Safety Evaluation**

Tests whether the model follows the project's customer-support safety requirements, including avoiding requests for sensitive credentials.

Result:

```text
Safety Pass Rate: 100%
```

**LLM-as-a-Judge**

A separate evaluator scores generated responses on:

```text
Correctness
Relevance
Helpfulness
Tone
Safety
```

Across the 5 evaluated prompts, the judge consistently gave **5/5 for tone and safety**.

**Base vs Fine-Tuned Comparison**

The same customer-support prompts are evaluated using both:

```text
Qwen3-0.6B Base
        vs
Qwen3-0.6B + LoRA Adapter
```

This makes it possible to inspect whether fine-tuning changed the model's behavior for the target domain.

---

## Safety-Focused Behavior

The training objective includes customer-support safety requirements such as avoiding requests for:

* Passwords
* OTPs
* CVVs
* Full payment-card numbers

The current safety evaluation achieved:

> **100% safety pass rate**

This is a project-specific evaluation result, not a claim of universal model safety.

---

## Example Behavior

### Order Tracking

**User**

```text
Where is my order?
```

**Fine-Tuned Model**

```text
Your order is currently in the system.
If you need further assistance, please contact us.
```

### Damaged Item

**User**

```text
I received a damaged item. What are my options?
```

**Fine-Tuned Model**

```text
I'll help you with the options. Please provide the product number.

Options:
Return, exchange, or refund.
```

### Failed Payment

**User**

```text
My payment failed. What should I do?
```

**Fine-Tuned Model**

```text
Your payment failed. Please check your account details.
If the issue persists, contact customer support for assistance.
```

These examples demonstrate the model's target behavior: **concise, polite, domain-specific customer-support responses with safety constraints**.

---

## Project Architecture

```text
customer_support_qlora/
│
├── data/
│   ├── raw/
│   │   └── customer_support_synthetic.jsonl
│   ├── processed/
│   │   ├── train.jsonl
│   │   └── eval.jsonl
│   ├── evaluation/
│   └── safety/
│
├── notebooks/
│   └── qlora_training.ipynb
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── validation.py
│   │
│   ├── training/
│   │   └── train.py
│   │
│   └── evaluation/
│       ├── base_vs_finetuned.py
│       ├── llm_judge.py
│       ├── qualitative_eval.py
│       ├── rouge_eval.py
│       └── safety_eval.py
│
├── configs/
│   └── training.yaml
│
├── tests/
│   ├── test_validation.py
│   └── test_safety.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Training Configuration

```yaml
model:
  name: Qwen/Qwen3-0.6B

quantization:
  bits: 4
  type: nf4
  double_quantization: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05

training:
  epochs: 2
  learning_rate: 0.0002
```

The actual experiment used **96 training examples and 24 evaluation examples**.

---

## Tech Stack

**LLM / Fine-Tuning**

* Qwen3-0.6B
* Hugging Face Transformers
* PEFT
* LoRA
* QLoRA
* BitsAndBytes
* TRL / SFT

**ML / Deep Learning**

* PyTorch
* Supervised Fine-Tuning
* 4-bit NF4 Quantization

**Evaluation**

* Evaluation Loss
* ROUGE
* LLM-as-a-Judge
* Safety Evaluation
* Qualitative Evaluation
* Base vs Fine-Tuned Comparison

**Engineering**

* Python
* YAML configuration
* Modular project structure
* Automated validation tests
* Git/GitHub

---

## Reproducibility

### 1. Clone

```bash
git clone <repository-url>
cd customer_support_qlora
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the training notebook

Open:

```text
notebooks/qlora_training.ipynb
```

Run the notebook on a CUDA-enabled GPU environment.

### 4. Run evaluation

The notebook evaluates:

```text
Evaluation Loss
ROUGE
Safety
Qualitative Examples
Base vs Fine-Tuned
LLM-as-a-Judge
```

---

## What I Learned

This project demonstrates practical understanding of:

* Why **LoRA** is preferred over full fine-tuning in resource-constrained environments.
* How **QLoRA** combines quantization with parameter-efficient fine-tuning.
* How **NF4 4-bit quantization** reduces memory requirements.
* How to select LoRA target modules in transformer architectures.
* How to prepare conversational datasets for SFT.
* Why held-out evaluation is necessary after training.
* Why **ROUGE alone is insufficient** for evaluating LLM responses.
* How to evaluate **safety separately from response quality**.
* How to compare a base model with its fine-tuned version.
* How to structure an LLM project into reusable training and evaluation modules.

---

## Limitations

This is a focused portfolio experiment rather than a production-scale customer-support model.

The evaluation dataset contains **24 examples**, and the LLM-as-a-Judge evaluation covers **5 representative prompts**, so the reported results should be interpreted as experimental evidence rather than statistically generalizable benchmarks.

The safety score represents the project's predefined safety checks and does not guarantee complete safety against all possible attacks or sensitive-information scenarios.

---

## Future Improvements

* Expand the customer-support dataset substantially.
* Add larger and more diverse evaluation sets.
* Add automated regression testing for model responses.
* Add stronger adversarial safety and prompt-injection evaluations.
* Compare multiple LoRA configurations.
* Benchmark additional base models.
* Add experiment tracking with MLflow/W&B.
* Deploy the adapter behind a FastAPI inference service.

---

## Author

**Md Shoaib Akhtar**

AI / Generative AI Engineer | B.Tech AI & Data Science

Focused on **Generative AI, LLM Fine-Tuning, RAG, Agentic AI, and Production AI Systems**.
