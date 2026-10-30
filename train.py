import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Load environment variables from .env file
load_dotenv()

# Your Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")

# Log in explicitly via Python
login(token=HF_TOKEN)

print("--- 1. Preparing Dataset ---")
raw_data = {
    "prompt": [
        "Convert this text to a JSON summary: John is 30 years old and lives in New York.",
        "Convert this text to a JSON summary: Sara is 25 years old and lives in London.",
        "Convert this text to a JSON summary: Michael is 42 years old and lives in Chicago.",
        "Convert this text to a JSON summary: Emily is 29 years old and lives in Seattle.",
        "Convert this text to a JSON summary: David is 35 years old and lives in Boston."
    ],
    "completion": [
        '{"name": "John", "age": 30, "city": "New York"}',
        '{"name": "Sara", "age": 25, "city": "London"}',
        '{"name": "Michael", "age": 42, "city": "Chicago"}',
        '{"name": "Emily", "age": 29, "city": "Seattle"}',
        '{"name": "David", "age": 35, "city": "Boston"}'
    ]
}

dataset = Dataset.from_dict({
    "text": [f"### Prompt:\n{p}\n\n### Response:\n{c}" for p, c in zip(raw_data["prompt"], raw_data["completion"])]
})

print("--- 2. Loading Base Model & Tokenizer ---")
model_id = "HuggingFaceTB/SmolLM2-360M"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model onto CPU...")
# Removed device_map="auto" to prevent meta-device tensor issues on CPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=HF_TOKEN,
    torch_dtype=torch.float32
)
model.to("cpu")

print("--- 3. Configuring LoRA ---")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("--- 4. Starting Training ---")
training_args = SFTConfig(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=1,
    num_train_epochs=5,
    max_length=256,
    dataset_text_field="text",
    use_cpu=True,
    fp16=False,
    bf16=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
)

trainer.train()

print("--- 5. Saving LoRA Adapter ---")
trainer.model.save_pretrained("./lora_adapter")
print("Training complete! Adapter saved to ./lora_adapter")