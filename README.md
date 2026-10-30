#  Fine-Tuning an LLM with LoRA (Low-Rank Adaptation)

This repository contains the complete code, configuration, and evaluation for fine-tuning an open-source small language model (**SmolLM2-360M**) on a custom structured instruction dataset using Hugging Face **PEFT** and **TRL**.

---

##  Project Overview
- **Task:** Instruction fine-tuning a model to convert plain-text personal descriptions into a strict, clean JSON summary format (`{"name": "...", "age": ..., "city": "..."}`).
- **Technique:** LoRA (Low-Rank Adaptation), updating only a fraction of the parameter weights to make training fast and resource-efficient.

---

##  Training Configuration
- **Base Model:** `HuggingFaceTB/SmolLM2-360M`
- **Training Library:** Hugging Face `trl` (`SFTTrainer` & `SFTConfig`)
- **PEFT Library:** `peft` (`LoraConfig`)
- **LoRA Rank ($r$):** 16
- **LoRA Alpha:** 32
- **Target Modules:** `q_proj`, `v_proj`, `k_proj`, `o_proj`
- **Learning Rate:** $2 \times 10^{-4}$
- **Epochs:** 5
- **Batch Size:** 2 (with gradient accumulation steps = 2)

---

##  Training Loss & Progress
The model successfully ran for 5 epochs on CPU/GPU, with the cross-entropy training loss steadily decreasing across iterations as it learned the target JSON output pattern.

---

##  Before / After Comparison

**Test Prompt:**  
`Convert this text to a JSON summary: Alex is 22 years old and lives in Berlin.`

###  Before (Base Model Output - Without Fine-Tuning)
> *The base model outputs conversational prose or struggles to format the raw text cleanly into valid JSON structure without prompt engineering guidance.*

###  After (Fine-Tuned Model Output - With LoRA Adapter)
> ```json
> {"name": "Alex", "age": 22, "city": "Berlin"}
> ```

---

##  Repository Structure
- `train.py` — The core script used to load the dataset, configure LoRA, and execute training.
- `inference.py` — The evaluation script used to load the base model vs. the fine-tuned LoRA adapter side-by-side.
- `./lora_adapter/` — Saved LoRA adapter weights (`adapter_model.safetensors`, `adapter_config.json`).
