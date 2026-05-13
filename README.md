# NebuDuck — Cybersecurity Mentor Chatbot

NebuDuck is a fine-tuned AI model that acts as a cybersecurity mentor. Instead of just giving you the answer, it guides you through problems using hints and questions. It won't hand out exploit code or walk you through attacks step by step — it's built to help you actually learn.

---

## What it does

You ask it a cybersecurity question, it gives you a nudge in the right direction. Think of it like a mentor who knows the answer but wants you to figure it out yourself.

The model follows these rules no matter what:
- No full exploits
- No direct answers
- Hints only
- Keep it short

---

## How it was built

The base model is Mistral 7B Instruct v0.2, running in 4-bit quantization to keep memory usage low. On top of that, LoRA adapters were trained on a custom cybersecurity Q&A dataset.

| Thing | What it is |
|---|---|
| Base model | mistral-7b-instruct-v0.2 |
| Quantization | 4-bit via bitsandbytes |
| Fine-tuning | LoRA via PEFT |
| Training framework | Unsloth + HuggingFace TRL |
| LoRA rank | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

---

## Dataset

The training data is a JSON file with input/output pairs like this:

```json
[
  {
    "input": "What is SQL injection?",
    "output": "Think about what happens when user input goes directly into a SQL query without any checks. What could someone do if they controlled part of that query?"
  }
]
```

Before training, each record gets wrapped in a prompt template that includes the system instructions, the user question, and the expected response.

---

## Training setup

Nothing too fancy here. Trained for 3 epochs on a Kaggle GPU using these settings:

| Setting | Value |
|---|---|
| Epochs | 3 |
| Learning rate | 2e-4 |
| Batch size | 2 |
| Gradient accumulation | 4 steps |
| Optimizer | AdamW 8-bit |
| Scheduler | Cosine |
| Max sequence length | 2048 |

---

## How to run it

Install the dependencies first:

```bash
pip install --upgrade huggingface_hub
pip install unsloth_zoo
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
pip install peft accelerate bitsandbytes datasets trl
```

Load the model:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./nebduck_lora",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

Ask it something:

```python
def ask_nebduck(question):
    prompt = f"### System:\nYou are NebuDuck, a cybersecurity mentor. Give hints only, no full answers.\n\n### User:\n{question}\n\n### Assistant:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9, do_sample=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("### Assistant:")[-1].strip()

print(ask_nebduck("How does a buffer overflow work?"))
```

---

## Example

Input:
> What is SQL injection?

NebuDuck:
> Think about what happens when user input gets embedded directly into a database query. What could go wrong if someone inserted their own SQL into it? Consider how quotes and logical operators might change what the query actually does.

---

## Limitations

- It won't give you full exploits or step-by-step attack guides, that's by design
- Quality depends heavily on the training data
- Needs a GPU to run properly (at least 8GB VRAM)

---

## Libraries used

- Unsloth — fast LoRA fine-tuning
- HuggingFace TRL — SFT trainer
- PEFT — LoRA adapters
- bitsandbytes — 4-bit quantization
- Mistral AI — base model
