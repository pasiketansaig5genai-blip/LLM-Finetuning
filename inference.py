import os
from dotenv import load_dotenv
# Load environment variables from .env file
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

load_dotenv()
# Your Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

model_id = "HuggingFaceTB/SmolLM2-360M"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# Test Prompt
test_prompt = "Convert this text to a JSON summary: Alex is 22 years old and lives in Berlin."
formatted_input = f"### Prompt:\n{test_prompt}\n\n### Response:\n"

inputs = tokenizer(formatted_input, return_tensors="pt")

# 1. Test Base Model (Without Fine-Tuning)
print("="*50)
print("BASE MODEL OUTPUT (Without Fine-Tuning):")
print("="*50)
base_model = AutoModelForCausalLM.from_pretrained(model_id, token=HF_TOKEN, torch_dtype=torch.float32)
base_outputs = base_model.generate(**inputs, max_new_tokens=40, temperature=0.1, do_sample=True)
print(tokenizer.decode(base_outputs[0], skip_special_tokens=True))

# 2. Test Fine-Tuned Model (With LoRA Adapter)
print("\n" + "="*50)
print("FINE-TUNED MODEL OUTPUT (With LoRA Adapter):")
print("="*50)
ft_base_model = AutoModelForCausalLM.from_pretrained(model_id, token=HF_TOKEN, torch_dtype=torch.float32)
ft_model = PeftModel.from_pretrained(ft_base_model, "./lora_adapter")
ft_outputs = ft_model.generate(**inputs, max_new_tokens=40, temperature=0.1, do_sample=True)
print(tokenizer.decode(ft_outputs[0], skip_special_tokens=True))