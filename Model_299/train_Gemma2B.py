import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Configuration & Paths
MODEL_ID = "google/gemma-4-E2B"
DATA_DIR = "data/vashantor010"
DIALECTS = ["Barishal", "Chittagong", "Mymensingh", "Noakhali", "Sylhet"]
OUTPUT_DIR = "./fine_tuned_gemma2b"

# 2. Data Preparation


def load_and_combine_data():
    all_rows = []
    for dialect in DIALECTS:
        for split in ["Test", "Validation"]:
            # Construct filename based on your folder structure
            file_path = os.path.join(
                DATA_DIR, f"{dialect} {split} Translation.csv")

            if os.path.exists(file_path):
                # Using the first 100 rows as requested
                df = pd.read_csv(file_path).head(100)

                # Format into Gemma 4 chat template
                for _, row in df.iterrows():
                    # Assuming columns are 'dialect' and 'standard'
                    messages = [
                        {"role": "user",
                            "content": f"Translate this dialect to standard Bengali: {row['dialect']}"},
                        {"role": "assistant", "content": row['standard']}
                    ]
                    all_rows.append({"messages": messages})
            else:
                print(f"Warning: File not found {file_path}")

    return Dataset.from_list(all_rows)


train_dataset = load_and_combine_data()

# 3. Model Loading with 4-bit Quantization (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for PEFT training
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "o_proj", "k_proj",
                    "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    bf16=True,  # Use bf16 for Gemma 4 stability
    optim="paged_adamw_32bit",
    remove_unused_columns=False  # Required for multimodal Gemma 4 processing
)

# 5. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    peft_config=lora_config,
    dataset_text_field="messages",  # TRL handles chat template automatically
)

# Start Fine-tuning
trainer.train()

# 6. Save Model
model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
processor.save_pretrained(OUTPUT_DIR)
print("Fine-tuning complete!")
