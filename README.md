# QLoRA Customer Support LLM

> **Parameter-Efficient Fine-Tuning of Qwen3-0.6B for E-Commerce Customer Support**

An end-to-end **LLM fine-tuning project** demonstrating supervised fine-tuning using **QLoRA (4-bit quantization + LoRA/PEFT)** on a conversational customer-support dataset.

The project is designed to demonstrate the complete fine-tuning workflow — from dataset preparation and validation to training, evaluation, adapter saving, and inference — using a GPU-enabled Google Colab environment.

---

## 🚀 Project Overview

The goal is to adapt a small open-source language model to follow a consistent **e-commerce customer-support style**.

The model is fine-tuned on conversational examples covering:

* Order tracking
* Delayed deliveries
* Returns & exchanges
* Refunds
* Damaged/wrong items
* Payment issues
* Coupons
* Account support
* Basic payment/security safety

### Pipeline

```text
Customer Support Dataset
          │
          ▼
Data Validation & Formatting
          │
          ▼
Chat Template
          │
          ▼
Qwen3-0.6B
          │
          ▼
4-bit Quantization
          │
          ▼
LoRA / PEFT
          │
          ▼
Supervised Fine-Tuning
          │
          ▼
LoRA Adapter
       ┌──┴──┐
       ▼     ▼
 Evaluation  Inference
```

---

## 🧠 Why Fine-Tuning?

Fine-tuning and RAG solve different problems.

**Fine-tuning** is useful when we want a model to learn:

* Response style
* Task-specific behavior
* Consistent output patterns
* Domain-specific conversational patterns

**RAG** is better for dynamic knowledge such as:

* Current product information
* Frequently changing policies
* Live inventory
* Order status
* Internal documentation

For a production customer-support system, **fine-tuning + RAG** could be used together.

---

## ⚡ Why QLoRA?

Full fine-tuning updates the parameters of the entire model and can require substantial GPU memory.

QLoRA reduces the memory requirement by:

1. Loading the base model in **4-bit precision**
2. Keeping the base model largely frozen
3. Training lightweight **LoRA adapters**

```text
                 Qwen3-0.6B
                /          \
               /            \
       Frozen Base       LoRA Adapter
       Parameters       Trainable
                            │
                            ▼
                      Fine-Tuned
                       Behavior
```

This makes experimentation possible on relatively limited hardware such as a **Google Colab GPU**.

---

## 🛠️ Tech Stack

| Component     | Technology                   |
| ------------- | ---------------------------- |
| Base Model    | Qwen3-0.6B                   |
| Fine-Tuning   | Supervised Fine-Tuning (SFT) |
| PEFT          | LoRA                         |
| Quantization  | 4-bit NF4                    |
| Training      | Hugging Face TRL             |
| Model Library | Transformers                 |
| Dataset       | Hugging Face Datasets        |
| Evaluation    | ROUGE + custom safety checks |
| Hardware      | Google Colab GPU             |
| Language      | Python                       |

---

## 📁 Project Structure

```text
qlora-customer-support-llm/
│
├── data/
│   ├── raw/
│   │   └── customer_support_synthetic.jsonl
│   │
│   └── processed/
│       ├── train.jsonl
│       └── eval.jsonl
│
├── notebooks/
│   └── 01_qlora_training.ipynb
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
│       ├── metrics.py
│       └── safety.py
│
├── configs/
│   └── training.yaml
│
├── outputs/
│   ├── adapter/
│   └── evaluation/
│
├── tests/
│   └── test_validation.py
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Dataset

For this portfolio prototype, the project uses a **synthetic conversational customer-support dataset**.

Each example follows a chat-based structure:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a professional e-commerce customer support assistant."
    },
    {
      "role": "user",
      "content": "My package is late."
    },
    {
      "role": "assistant",
      "content": "I'm sorry about the delay. Please provide your order ID..."
    }
  ]
}
```

The dataset is separated into:

```text
train.jsonl
eval.jsonl
```

so that evaluation is performed on held-out examples.

> **Note:** The dataset is intentionally small and synthetic because this repository demonstrates the fine-tuning engineering workflow rather than production-scale model training.

For a production implementation, I would use a larger approved dataset with **PII removal, data-quality filtering, human review, deduplication, and stronger evaluation**.

---

## 🔬 Training Configuration

The project uses:

```text
Base Model       : Qwen3-0.6B
Method           : SFT + LoRA
Quantization     : 4-bit NF4
LoRA Rank        : 16
LoRA Alpha       : 32
LoRA Dropout     : 0.05
Learning Rate    : 2e-4
Epochs           : 2
Batch Size       : 2
Gradient Accum.  : 4
Max Sequence     : 512
```

These settings are intentionally kept lightweight so the experiment can run on a Colab GPU.

---

## 📈 Evaluation

The project evaluates the fine-tuned model using:

### 1. ROUGE

Used as an automatic text-overlap metric against held-out reference responses.

### 2. Safety Checks

Basic checks look for unsafe handling of sensitive information such as:

```text
Password
OTP
CVV
Full payment-card numbers
```

### 3. Qualitative Evaluation

The notebook also generates responses for previously unseen customer-support questions so that the behavior can be inspected manually.

---

## ▶️ Running the Project in Google Colab

### 1. Clone or download the repository

Upload the project to your Colab environment.

### 2. Enable GPU

```text
Colab
 → Runtime
 → Change runtime type
 → Hardware accelerator
 → GPU
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

```text
notebooks/01_qlora_training.ipynb
```

Run the notebook from top to bottom.

The notebook performs:

```text
Dataset loading
      ↓
Data formatting
      ↓
Tokenizer setup
      ↓
4-bit model loading
      ↓
LoRA configuration
      ↓
SFT training
      ↓
Evaluation
      ↓
Inference
      ↓
Adapter saving
```

---

## 💾 Output

After training, the LoRA adapter is saved under:

```text
outputs/adapter/
```

The evaluation report is saved under:

```text
outputs/evaluation/evaluation_report.json
```

The adapter contains the learned LoRA parameters rather than a completely duplicated copy of the base model.

For inference, the compatible base model and trained adapter are loaded together.

---

## 🎯 Key Learning Outcomes

This project demonstrates understanding of:

* LLM fine-tuning
* Supervised Fine-Tuning
* Causal Language Models
* LoRA
* PEFT
* QLoRA
* 4-bit quantization
* Chat templates
* Dataset formatting
* Train/evaluation splitting
* Training hyperparameters
* GPU memory optimization
* Adapter-based model saving
* Automatic evaluation
* Basic LLM safety evaluation
* Fine-tuning vs RAG trade-offs

---

## 🏗️ Production Considerations

This repository is a **portfolio prototype**, not a production customer-support system.

For production, I would additionally consider:

* Larger high-quality domain dataset
* PII detection and removal
* Human annotation/review
* Dataset deduplication
* Stronger offline evaluation
* LLM-as-a-judge evaluation
* Regression test set
* RAG for dynamic business knowledge
* Model/version tracking
* Experiment tracking
* Monitoring
* Guardrails
* Rate limiting
* Authentication and authorization
* Quantized inference/deployment optimization

---

## 💡 Key Design Decision

### Why not full fine-tuning?

Because the objective is to demonstrate **parameter-efficient adaptation** while keeping GPU and memory requirements manageable.

### Why not train the entire Qwen model?

For this use case, updating all base parameters would be unnecessarily expensive for a small domain dataset.

### Why keep the adapter separate?

LoRA adapters are small and can be maintained independently from the base model, making experimentation and deployment more practical.

---

## 🗣️ Interview Explanation

A concise way to explain the project:

> "I built an end-to-end customer-support LLM fine-tuning pipeline using Qwen3-0.6B. I used supervised fine-tuning with LoRA and 4-bit quantization through QLoRA to make training feasible on a Colab GPU. I separated the dataset into training and held-out evaluation sets, applied the model's chat template, trained only the LoRA adapters, evaluated the resulting behavior, and saved the adapter separately from the base model. For production, I would combine the fine-tuned behavior with RAG for dynamic customer-support knowledge."

---

## ⚠️ Limitations

This project intentionally uses a small synthetic dataset.

Therefore:

* It should not be considered production-ready.
* Evaluation scores should not be interpreted as proof of broad model capability.
* The model may overfit to patterns present in the small dataset.
* Real-world deployment would require significantly more data and evaluation.

The primary objective is to demonstrate a **complete, understandable and reproducible fine-tuning workflow**.

---

## 📌 Future Improvements

* Expand the dataset with real approved support conversations
* Add data-quality and PII filtering
* Compare **base model vs fine-tuned model**
* Compare LoRA ranks
* Perform hyperparameter experiments
* Add LLM-as-a-judge evaluation
* Add experiment tracking with MLflow/W&B
* Build a FastAPI inference API
* Add RAG for dynamic support knowledge
* Deploy the adapter-backed model

---

## 👨‍💻 Project Focus

**GenAI Engineering • LLM Fine-Tuning • QLoRA • PEFT • LLM Evaluation**
