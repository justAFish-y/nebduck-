import os
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

print("CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Enable GPU in Kaggle: Settings > Accelerator > GPU T4 x2")
print("GPU:", torch.cuda.get_device_name(0))

DATASET_DIR = "/kaggle/input/datasets/nebuduck/dataset"
json_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".json")]
if not json_files:
    raise FileNotFoundError("No JSON file found in dataset folder")
DATASET_PATH = os.path.join(DATASET_DIR, json_files[0])
print("Using:", DATASET_PATH)
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
raw_data = [
    r for r in raw_data
    if isinstance(r, dict)
    and isinstance(r.get("input"), str)
    and isinstance(r.get("output"), str)
]
print("Clean dataset size:", len(raw_data))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

NEBDUCK_SYSTEM = """
You are NebuDuck, a cybersecurity mentor.
Rules:
- No full exploits
- No direct answers
- Give hints only
- Be concise
"""
EOS = tokenizer.eos_token

def format_prompt(x):
    return {
        "text": (
            f"### System:\n{NEBDUCK_SYSTEM}\n\n"
            f"### User:\n{x['input']}\n\n"
            f"### Assistant:\n{x['output']}{EOS}"
        )
    }

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
print("Formatted dataset:", len(dataset))

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=2048,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="/kaggle/working/nebduck",
        save_strategy="epoch",
        report_to="none",
        dataset_num_proc=2,
        packing=False,
    ),
)

print("Training starting...")
trainer_stats = trainer.train()
print("Training done. Final loss:", trainer_stats.training_loss)

SAVE_PATH = "/kaggle/working/nebduck_lora"
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print("Saved to:", SAVE_PATH)

FastLanguageModel.for_inference(model)

def ask_nebduck(q):
    prompt = (
        f"### System:\n{NEBDUCK_SYSTEM}\n\n"
        f"### User:\n{q}\n\n"
        f"### Assistant:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("### Assistant:")[-1].strip()

print(ask_nebduck("What is SQL injection?"))
