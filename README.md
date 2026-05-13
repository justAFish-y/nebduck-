# 🦆 NebuDuck — Cybersecurity Mentor Chatbot

A fine-tuned large language model trained to act as a cybersecurity mentor. NebuDuck guides users through security concepts using hints and Socratic reasoning — never giving direct exploits or full answers.

---

## 🧠 What Is NebuDuck?

NebuDuck is a fine-tuned version of **Mistral 7B Instruct v0.2**, trained on a custom cybersecurity Q&A dataset. It is designed to behave like a responsible security mentor: it explains concepts, gives hints, and encourages thinking — without handing out complete attack code or bypasses.

**Core behavior rules:**
- No full exploits
- No direct answers
- Hints and guidance only
- Concise and educational responses

---

## 🏗️ Architecture

| Component | Detail |
|---|---|
| Base Model | `mistral-7b-instruct-v0.2` |
| Quantization | 4-bit (bitsandbytes NF4) |
| Fine-tuning Method | LoRA (Low-Rank Adaptation) via PEFT |
| Training Framework | Unsloth + HuggingFace TRL |
| LoRA Rank | 16 |
| LoRA Alpha | 16 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

LoRA adapters are trained on top of a frozen 4-bit quantized base model, making the fine-tuning process memory-efficient and fast on a single GPU.

---

## 📂 Dataset Format

The training dataset is a JSON file consisting of input/output pairs:

```json
[
  {
    "input": "What is SQL injection?",
    "output": "Think about what happens when user input is directly inserted into a SQL query without sanitization. What could an attacker do if they controlled part of that query?"
  },
  ...
]
```

Each record was filtered to ensure both `input` and `output` are non-empty strings. The dataset was formatted using the following prompt template before training:

```
### System:
You are NebuDuck, a cybersecurity mentor.
Rules:
- No full exploits
- No direct answers
- Give hints only
- Be concise

### User:
{input}

### Assistant:
{output}<eos>
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Learning Rate | 2e-4 |
| Batch Size (per device) | 2 |
| Gradient Accumulation Steps | 4 |
| Effective Batch Size | 8 |
| Optimizer | AdamW 8-bit |
| LR Scheduler | Cosine |
| Max Sequence Length | 2048 |
| Precision | bf16 (if supported), else fp16 |
| Gradient Checkpointing | Unsloth (memory optimized) |

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install --upgrade huggingface_hub
pip install unsloth_zoo
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
pip install peft accelerate bitsandbytes datasets trl
```

### 2. Load the Model

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./nebduck_lora",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

### 3. Run Inference

```python
SYSTEM = """
You are NebuDuck, a cybersecurity mentor.
Rules:
- No full exploits
- No direct answers
- Give hints only
- Be concise
"""

def ask_nebduck(question):
    prompt = f"### System:\n{SYSTEM}\n\n### User:\n{question}\n\n### Assistant:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("### Assistant:")[-1].strip()

print(ask_nebduck("How does a buffer overflow work?"))
```

---

## 💬 Example Output

**Input:**
> What is SQL injection?

**NebuDuck:**
> Consider what happens when user-supplied input is embedded directly into a database query. What could go wrong if an attacker inserted their own SQL syntax? Try thinking about how quotes and logical operators might change the query's meaning.

---

## ⚠️ Limitations

- NebuDuck will not provide complete exploit code or step-by-step attack walkthroughs
- Responses are hint-based and may require follow-up questions
- Model performance is constrained by the size and quality of the training dataset
- Running inference requires a CUDA-capable GPU (minimum 8GB VRAM recommended)

---

## 🛠️ Built With

- [Unsloth](https://github.com/unslothai/unsloth) — fast LoRA fine-tuning
- [HuggingFace TRL](https://github.com/huggingface/trl) — SFT trainer
- [PEFT](https://github.com/huggingface/peft) — LoRA adapter support
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit quantization
- [Mistral AI](https://mistral.ai/) — base model
