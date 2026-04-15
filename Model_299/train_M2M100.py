import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

# 1. Load the cleaned data

train_df = pd.read_csv('./data/processed/cleaned_train.csv')
val_df = pd.read_csv('./data/processed/cleaned_val.csv')

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)

tokenizer.src_lang = "bn"
tokenizer.tgt_lang = "bn"

# 2. Preprocess function

def preprocess_function(examples):
    inputs = examples["text"]       
    # Standard Bengali
    targets = examples["standard"]

    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# 3. Training Arguments 
training_args = Seq2SeqTrainingArguments(
    output_dir="./fine_tuned_m2m100",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    logging_steps=10,
    push_to_hub=False
)

# 4. Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,  # Updated from 'tokenizer'
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

print("Starting training...")
trainer.train()

# 5. Save the final "Standard Bengali" Brain
trainer.save_model("./fine_tuned_m2m100")
tokenizer.save_pretrained("./fine_tuned_m2m100")
print("Success! Translator model saved.")
