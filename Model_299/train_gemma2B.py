import os
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

# CONFIGURATION

class Config:
    # MODEL ID
    MODEL_ID = "google/gemma-4-E2B"
    MAX_SEQ_LENGTH = 250
    TRAIN_PATH = "./data/processed/cleaned_train.csv"
    VAL_PATH = "./data/processed/cleaned_val.csv"
    OUTPUT_DIR = "./gemma4_dialect_results"
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]

    # Training Hyperparameters

    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    GRAD_ACCUM = 4
    LEARNING_RATE = 2e-4
    SEED = 42

# Instruction format
INSTRUCTION = "Convert the following Bengali dialect to standard Bengali:\n"
RESPONSE_TEMPLATE = "### Standard Bengali:\n"

# DATA PREPARATION

def formatting_prompts_func(example):
    """
    Creates a prompt structure that clearly separates dialect input from standard output.
    """
    output_texts = []
    for i in range(len(example['text'])):
        text = (
            f"### Dialect Input:\n{example['text'][i]}\n\n"
            f"### Instruction:\n{INSTRUCTION}\n\n"
            f"{RESPONSE_TEMPLATE}{example['standard'][i]}"
        )
        output_texts.append(text)
    return output_texts


def main():
    set_seed(Config.SEED)

    # 1. Load Data
    train_df = pd.read_csv(Config.TRAIN_PATH).dropna(
        subset=['text', 'standard'])
    val_df = pd.read_csv(Config.VAL_PATH).dropna(subset=['text', 'standard'])

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df)
    })

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_ID)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Model Loading 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True 
    )

    model = prepare_model_for_kbit_training(model)

    # 4. LoRA SETUP FOR GEMMA-4

    target_modules = [
        "q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear",
        "gate_proj.linear", "up_proj.linear", "down_proj.linear"
    ]

    peft_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=Config.LORA_DROPOUT,
        task_type="CAUSAL_LM",
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 5. Training Arguments
    sft_config = SFTConfig(
        output_dir=Config.OUTPUT_DIR,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=True,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUM,
        learning_rate=Config.LEARNING_RATE,
        num_train_epochs=Config.NUM_EPOCHS,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    # 6. Data Collator and Trainer
    response_template_ids = tokenizer.encode(
        RESPONSE_TEMPLATE, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        args=sft_config,
    )

    print(
        f"Starting Fine-tuning for {Config.MODEL_ID} with ClippableLinear Fix...")
    trainer.train()

    # Save
    final_output = os.path.join(Config.OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)
    print(f"Done! Model saved to {final_output}")


if __name__ == "__main__":
    main()
